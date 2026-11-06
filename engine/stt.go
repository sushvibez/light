package engine

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"slices"
	"strings"
	"time"

	"light/audio"
	"light/dtype"
	"light/hub"
	"light/models"
	"light/tensor"
	"light/tokenizer"
)

// Transcriber is a loaded speech-to-text checkpoint with its tokenizer and
// mel filterbank: Whisper (openai/whisper-*) or Qwen3-ASR
// (Qwen/Qwen3-ASR-*-hf). Exactly one of Model / Qwen is set.
type Transcriber struct {
	Spec     Spec
	Model    *models.Whisper  // Family "whisper"
	Qwen     *models.Qwen3ASR // Family "qwen3_asr"
	Tok      *tokenizer.BPE
	Filters  [][]float32
	LoadTime time.Duration
}

// TranscribeResult is one transcription with timings.
type TranscribeResult struct {
	Text      string  `json:"text"`
	Language  string  `json:"language,omitempty"`
	AudioSec  float64 `json:"audio_sec"`
	MelMS     float64 `json:"mel_ms"`
	EncodeMS  float64 `json:"encode_ms"`
	DecodeMS  float64 `json:"decode_ms"`
	Tokens    int     `json:"tokens"`
	Chunks    int     `json:"chunks"`
	RealTimeX float64 `json:"real_time_x"` // audio seconds per wall second
}

// LoadTranscriber loads a Whisper repo (openai/whisper-tiny … large-v3) or
// a Qwen3-ASR repo. Spec.DType picks the Qwen3-ASR decoder precision
// (bf16 default; its encoder is always F32).
func LoadTranscriber(spec Spec) (*Transcriber, error) {
	t0 := time.Now()
	dev, err := pickDevice(spec.Device)
	if err != nil {
		return nil, err
	}
	if spec.Repo == "" {
		return nil, errors.New("engine: transcriber spec needs Repo")
	}
	modelType, err := hfModelType(spec.Repo)
	if err != nil {
		return nil, err
	}
	tokPath, err := hub.Download(spec.Repo, "main", "tokenizer.json")
	if err != nil {
		return nil, err
	}
	f, err := os.Open(tokPath)
	if err != nil {
		return nil, err
	}
	tok, err := tokenizer.LoadBPE(f)
	f.Close()
	if err != nil {
		return nil, err
	}
	tr := &Transcriber{Spec: spec, Tok: tok}
	switch modelType {
	case "whisper":
		if tr.Model, err = models.LoadWhisperFromHF(spec.Repo, dev); err != nil {
			return nil, err
		}
		if tr.Filters, err = models.LoadWhisperMelFilters(spec.Repo); err != nil {
			return nil, err
		}
		tr.Spec.Family = "whisper"
	case "qwen3_asr":
		dt := dtype.BF16
		if strings.EqualFold(spec.DType, "f32") {
			dt = dtype.F32
		}
		if tr.Qwen, err = models.LoadQwen3ASRFromHF(spec.Repo, dev, dt); err != nil {
			return nil, err
		}
		tr.Filters = audio.MelFilterBank(tr.Qwen.Cfg.NumMelBins, audio.WhisperNFFT, audio.WhisperSampleRate, 0, 8000)
		tr.Spec.Family = "qwen3_asr"
	default:
		return nil, fmt.Errorf("engine: %s is model_type %q, not a speech model", spec.Repo, modelType)
	}
	tr.LoadTime = time.Since(t0)
	return tr, nil
}

// hfModelType reads model_type from a repo's config.json.
func hfModelType(repo string) (string, error) {
	path, err := hub.Download(repo, "main", "config.json")
	if err != nil {
		return "", err
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	var c struct {
		ModelType string `json:"model_type"`
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		return "", err
	}
	return c.ModelType, nil
}

// TranscribeWAV decodes a WAV file and transcribes it.
func (t *Transcriber) TranscribeWAV(path string, opts models.TranscribeOpts) (*TranscribeResult, error) {
	samples, rate, err := audio.ReadWAV(path)
	if err != nil {
		return nil, err
	}
	return t.Transcribe(samples, rate, opts)
}

// Transcribe runs the model over mono samples at any rate. Whisper cuts
// audio longer than 30 s into consecutive 30-second chunks whose
// transcripts are joined; Qwen3-ASR takes the whole clip in one pass
// (its encoder attends in 8-second windows and the decoder sees every
// audio token).
func (t *Transcriber) Transcribe(samples []float32, rate int, opts models.TranscribeOpts) (*TranscribeResult, error) {
	if t.Qwen != nil {
		return t.transcribeQwen(samples, rate, opts)
	}
	start := time.Now()
	s16 := audio.Resample(samples, rate, audio.WhisperSampleRate)
	res := &TranscribeResult{AudioSec: float64(len(s16)) / audio.WhisperSampleRate}
	chunk := audio.WhisperSampleRate * audio.WhisperChunkSec
	dev := t.Model.Device()
	var parts []string
	for off := 0; off < len(s16) || off == 0; off += chunk {
		end := min(off+chunk, len(s16))
		t0 := time.Now()
		mel, err := audio.LogMel(s16[off:end], t.Filters)
		if err != nil {
			return nil, err
		}
		res.MelMS += ms(time.Since(t0))
		melT, err := tensor.New(mel, []int{1, len(t.Filters), audio.WhisperFrames}, dev)
		if err != nil {
			return nil, err
		}
		t1 := time.Now()
		enc, err := t.Model.Encode(melT)
		if err != nil {
			return nil, err
		}
		res.EncodeMS += ms(time.Since(t1))
		t2 := time.Now()
		ids, err := t.Model.TranscribeEncoded(enc, opts)
		if err != nil {
			return nil, err
		}
		res.DecodeMS += ms(time.Since(t2))
		res.Tokens += len(ids)
		res.Chunks++
		ids32 := make([]int32, len(ids))
		for i, v := range ids {
			ids32[i] = int32(v)
		}
		parts = append(parts, t.Tok.Decode(ids32))
		if len(s16) == 0 {
			break
		}
	}
	res.Text = strings.TrimSpace(strings.Join(parts, ""))
	res.Language = opts.Language
	if wall := time.Since(start).Seconds(); wall > 0 {
		res.RealTimeX = res.AudioSec / wall
	}
	return res, nil
}

// qwen3ASRMinSamples mirrors the feature extractor's min_length.
const qwen3ASRMinSamples = 8000

// transcribeQwen is the Qwen3ASRProcessor pipeline: 16 kHz audio padded
// to ≥ 0.5 s, 128-bin log-mel zero-padded to whole 100-frame chunks,
// one encoder pass, then greedy decoding of "language <Name><asr_text>…".
func (t *Transcriber) transcribeQwen(samples []float32, rate int, opts models.TranscribeOpts) (*TranscribeResult, error) {
	start := time.Now()
	s16 := audio.Resample(samples, rate, audio.WhisperSampleRate)
	res := &TranscribeResult{AudioSec: float64(len(s16)) / audio.WhisperSampleRate, Chunks: 1}
	if len(s16) < qwen3ASRMinSamples {
		s16 = append(slices.Clone(s16), make([]float32, qwen3ASRMinSamples-len(s16))...)
	}
	t0 := time.Now()
	mel, frames, err := audio.LogMelFrames(s16, t.Filters)
	if err != nil {
		return nil, err
	}
	ch := t.Qwen.Cfg.ChunkFrames()
	padded := (frames + ch - 1) / ch * ch
	nMel := len(t.Filters)
	full := make([]float32, nMel*padded)
	for b := range nMel {
		copy(full[b*padded:], mel[b*frames:(b+1)*frames])
	}
	res.MelMS = ms(time.Since(t0))
	t1 := time.Now()
	enc, err := t.Qwen.Encode(full, frames)
	if err != nil {
		return nil, err
	}
	audioEmb, err := t.Qwen.Project(enc)
	if err != nil {
		return nil, err
	}
	res.EncodeMS = ms(time.Since(t1))

	var prefill []int64
	if name := qwen3ASRLanguageName(opts.Language); name != "" {
		for _, id := range t.Tok.Encode("language " + name) {
			prefill = append(prefill, int64(id))
		}
		prefill = append(prefill, t.Qwen.Cfg.ASRTextID)
	}
	var system []int64
	if opts.Prompt != "" {
		for _, id := range t.Tok.Encode(opts.Prompt) {
			system = append(system, int64(id))
		}
	}
	t2 := time.Now()
	ids, err := t.Qwen.TranscribeAudio(audioEmb, system, prefill, opts.MaxTokens)
	if err != nil {
		return nil, err
	}
	res.DecodeMS = ms(time.Since(t2))
	res.Tokens = len(ids)
	all := append(slices.Clone(prefill), ids...)
	res.Language, res.Text = t.parseQwenOutput(all)
	if wall := time.Since(start).Seconds(); wall > 0 {
		res.RealTimeX = res.AudioSec / wall
	}
	return res, nil
}

// parseQwenOutput splits "language <Name><asr_text>text" into its parts
// (language code when the name is known, else the raw name).
func (t *Transcriber) parseQwenOutput(ids []int64) (lang, text string) {
	split := slices.Index(ids, t.Qwen.Cfg.ASRTextID)
	if split < 0 {
		return "", strings.TrimSpace(t.Tok.Decode(toInt32(ids)))
	}
	head := strings.TrimSpace(t.Tok.Decode(toInt32(ids[:split])))
	name := strings.TrimSpace(strings.TrimPrefix(head, "language"))
	lang = name
	for code, n := range qwen3ASRLanguages {
		if strings.EqualFold(n, name) {
			lang = code
			break
		}
	}
	if strings.EqualFold(name, "none") {
		lang = ""
	}
	return lang, strings.TrimSpace(t.Tok.Decode(toInt32(ids[split+1:])))
}

func toInt32(ids []int64) []int32 {
	out := make([]int32, len(ids))
	for i, v := range ids {
		out[i] = int32(v)
	}
	return out
}

// qwen3ASRLanguages is the processor's LANGUAGE_CODE_TO_NAME table.
var qwen3ASRLanguages = map[string]string{
	"ar": "Arabic", "yue": "Cantonese", "zh": "Chinese", "cs": "Czech", "da": "Danish", "nl": "Dutch",
	"en": "English", "fil": "Filipino", "fi": "Finnish", "fr": "French", "de": "German", "el": "Greek",
	"hi": "Hindi", "hu": "Hungarian", "id": "Indonesian", "it": "Italian", "ja": "Japanese", "ko": "Korean",
	"mk": "Macedonian", "ms": "Malay", "fa": "Persian", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian",
	"ru": "Russian", "es": "Spanish", "sv": "Swedish", "th": "Thai", "tr": "Turkish", "vi": "Vietnamese",
}

// qwen3ASRLanguageName maps a code ("en") or name ("English") to the name
// the model was trained with; "" means auto-detect.
func qwen3ASRLanguageName(lang string) string {
	if lang == "" {
		return ""
	}
	if n, ok := qwen3ASRLanguages[strings.ToLower(lang)]; ok {
		return n
	}
	for _, n := range qwen3ASRLanguages {
		if strings.EqualFold(n, lang) {
			return n
		}
	}
	return lang
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

// Describe is a one-line summary.
func (t *Transcriber) Describe() string {
	n := 0
	if t.Model != nil {
		n = t.Model.ParamCount()
	} else if t.Qwen != nil {
		n = t.Qwen.ParamCount()
	}
	return fmt.Sprintf("%s (%.0fM params)", t.Spec.ID, float64(n)/1e6)
}
