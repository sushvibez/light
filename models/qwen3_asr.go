package models

import (
	"fmt"
	"math"

	"light/autograd"
	"light/device"
	"light/dtype"
	"light/nn"
	"light/ops"
	"light/tensor"
)

// Qwen3ASRConfig mirrors the audio_config block of Qwen/Qwen3-ASR-*-hf
// plus the special token ids the prompt needs.
type Qwen3ASRConfig struct {
	NumMelBins       int // 128
	DModel           int // 1024
	Layers           int // 24
	Heads            int // 16
	FFN              int // 4096
	DownsampleHidden int // 480 conv channels
	NWindow          int // 50: mel chunk = 2*NWindow frames
	NWindowInfer     int // 800: attention window in mel frames
	OutputDim        int // 2048 = LM hidden size
	ChunkPositions   int // 13 = post-conv frames per chunk (max_position_embeddings)

	AudioTokenID int64 // <|audio_pad|>
	AudioStartID int64 // <|audio_start|>
	AudioEndID   int64 // <|audio_end|>
	ASRTextID    int64 // <asr_text>
	EOS          []int64
}

// ChunkFrames is the mel-frame length of one conv chunk (2*NWindow).
func (c Qwen3ASRConfig) ChunkFrames() int { return 2 * c.NWindow }

// postConvLen is the frame count after the three (k=3, s=2, p=1) convs.
func postConvLen(n int) int {
	for range 3 {
		if n <= 0 {
			return 0
		}
		n = (n-1)/2 + 1
	}
	return n
}

// AudioTokens returns how many <|audio_pad|> tokens `validFrames` mel
// frames become (transformers' _get_feat_extract_output_lengths).
func (c Qwen3ASRConfig) AudioTokens(validFrames int) int {
	ch := c.ChunkFrames()
	return (validFrames/ch)*c.ChunkPositions + postConvLen(validFrames%ch)
}

type qwen3ASRAttn struct {
	Q, K, V, O *nn.Linear
}

type qwen3ASREncLayer struct {
	LN1  *nn.LayerNorm
	Attn *qwen3ASRAttn
	LN2  *nn.LayerNorm
	FC1  *nn.Linear
	FC2  *nn.Linear
}

// Qwen3ASR is the AuT audio encoder + projector + Qwen3 decoder of
// Qwen3-ASR. The encoder runs at F32; the decoder at LM.Cfg.WeightDType.
type Qwen3ASR struct {
	Cfg Qwen3ASRConfig

	Conv1W, Conv1B *tensor.Tensor // (C, 1, 3, 3)
	Conv2W, Conv2B *tensor.Tensor // (C, C, 3, 3)
	Conv3W, Conv3B *tensor.Tensor
	ConvOut        *nn.Linear     // (D, C*mel/8), no bias
	Pos            *tensor.Tensor // (ChunkPositions, D) sinusoids
	Layers         []*qwen3ASREncLayer
	LNPost         *nn.LayerNorm
	Proj1, Proj2   *nn.Linear // D→D (GELU) →OutputDim

	LM *Llama

	scale *tensor.Tensor
	dev   device.Device
}

// NewQwen3ASR allocates the encoder/projector (F32, uninitialised) around
// an already-built decoder.
func NewQwen3ASR(cfg Qwen3ASRConfig, lm *Llama, dev device.Device) (*Qwen3ASR, error) {
	D, C := cfg.DModel, cfg.DownsampleHidden
	if D%cfg.Heads != 0 {
		return nil, fmt.Errorf("qwen3_asr: d_model %d not divisible by %d heads", D, cfg.Heads)
	}
	if lm.Cfg.HiddenSize != cfg.OutputDim {
		return nil, fmt.Errorf("qwen3_asr: projector output %d != decoder hidden %d", cfg.OutputDim, lm.Cfg.HiddenSize)
	}
	m := &Qwen3ASR{Cfg: cfg, LM: lm, dev: dev}
	var err error
	z := func(dims ...int) *tensor.Tensor {
		if err != nil {
			return nil
		}
		var t *tensor.Tensor
		t, err = tensor.Zeros(dtype.F32, dims, dev)
		return t
	}
	m.Conv1W, m.Conv1B = z(C, 1, 3, 3), z(C)
	m.Conv2W, m.Conv2B = z(C, C, 3, 3), z(C)
	m.Conv3W, m.Conv3B = z(C, C, 3, 3), z(C)
	if err != nil {
		return nil, err
	}
	freq := cfg.NumMelBins
	for range 3 {
		freq = (freq-1)/2 + 1
	}
	lin := func(in, out int, bias bool) (*nn.Linear, error) {
		return nn.NewLinearUninit(in, out, bias, dtype.F32, dev)
	}
	if m.ConvOut, err = lin(C*freq, D, false); err != nil {
		return nil, err
	}
	if m.Pos, err = sinusoidPositions(cfg.ChunkPositions, D, dev); err != nil {
		return nil, err
	}
	ln := func() (*nn.LayerNorm, error) { return nn.NewLayerNorm(D, 1e-5, dev) }
	for range cfg.Layers {
		l := &qwen3ASREncLayer{Attn: &qwen3ASRAttn{}}
		if l.LN1, err = ln(); err != nil {
			return nil, err
		}
		for _, p := range []**nn.Linear{&l.Attn.Q, &l.Attn.K, &l.Attn.V, &l.Attn.O} {
			if *p, err = lin(D, D, true); err != nil {
				return nil, err
			}
		}
		if l.LN2, err = ln(); err != nil {
			return nil, err
		}
		if l.FC1, err = lin(D, cfg.FFN, true); err != nil {
			return nil, err
		}
		if l.FC2, err = lin(cfg.FFN, D, true); err != nil {
			return nil, err
		}
		m.Layers = append(m.Layers, l)
	}
	if m.LNPost, err = ln(); err != nil {
		return nil, err
	}
	if m.Proj1, err = lin(D, D, true); err != nil {
		return nil, err
	}
	if m.Proj2, err = lin(D, cfg.OutputDim, true); err != nil {
		return nil, err
	}
	hd := D / cfg.Heads
	if m.scale, err = tensor.New([]float32{float32(1 / math.Sqrt(float64(hd)))}, []int{1}, dev); err != nil {
		return nil, err
	}
	return m, nil
}

// sinusoidPositions is Qwen3ASR's SinusoidsPositionEmbedding: [sin | cos]
// over channels/2 timescales from 1 to 10000.
func sinusoidPositions(length, channels int, dev device.Device) (*tensor.Tensor, error) {
	half := channels / 2
	inc := math.Log(10000) / float64(half-1)
	data := make([]float32, length*channels)
	for p := range length {
		for i := range half {
			a := float64(p) * math.Exp(-inc*float64(i))
			data[p*channels+i] = float32(math.Sin(a))
			data[p*channels+half+i] = float32(math.Cos(a))
		}
	}
	return tensor.New(data, []int{length, channels}, dev)
}

// Device returns the device the weights live on.
func (m *Qwen3ASR) Device() device.Device { return m.dev }

// ParamCount counts encoder + projector + decoder weights.
func (m *Qwen3ASR) ParamCount() int {
	n := m.LM.ParamCount()
	for _, t := range m.encoderParams() {
		n += t.NumElements()
	}
	return n
}

func (m *Qwen3ASR) encoderParams() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.Conv1W, m.Conv1B, m.Conv2W, m.Conv2B, m.Conv3W, m.Conv3B, m.ConvOut.W}
	for _, l := range m.Layers {
		ps = append(ps, l.LN1.Gamma, l.LN1.Beta, l.LN2.Gamma, l.LN2.Beta, l.FC1.W, l.FC1.B, l.FC2.W, l.FC2.B,
			l.Attn.Q.W, l.Attn.Q.B, l.Attn.K.W, l.Attn.K.B, l.Attn.V.W, l.Attn.V.B, l.Attn.O.W, l.Attn.O.B)
	}
	return append(ps, m.LNPost.Gamma, m.LNPost.Beta, m.Proj1.W, m.Proj1.B, m.Proj2.W, m.Proj2.B)
}

// Encode runs the audio encoder. mel is a (NumMelBins × frames) row-major
// log-mel spectrogram (audio.LogMelFrames with the 128-bin filterbank),
// of which the first validFrames are real audio; frames must be a
// multiple of ChunkFrames (zero-pad to it). Returns the packed encoder
// output (S, DModel) after ln_post, S = AudioTokens(validFrames).
func (m *Qwen3ASR) Encode(mel []float32, validFrames int) (*tensor.Tensor, error) {
	cfg := m.Cfg
	nMel, ch := cfg.NumMelBins, cfg.ChunkFrames()
	if len(mel)%nMel != 0 {
		return nil, fmt.Errorf("Qwen3ASR.Encode: mel length %d not a multiple of %d bins", len(mel), nMel)
	}
	frames := len(mel) / nMel
	if frames%ch != 0 || frames == 0 {
		return nil, fmt.Errorf("Qwen3ASR.Encode: %d frames, want a non-zero multiple of %d", frames, ch)
	}
	if validFrames <= 0 || validFrames > frames {
		return nil, fmt.Errorf("Qwen3ASR.Encode: validFrames %d outside (0, %d]", validFrames, frames)
	}
	nChunks := frames / ch
	// (nChunks, 1, nMel, ch): chunk c, bin b, frame t ← mel[b*frames + c*ch + t].
	data := make([]float32, nChunks*nMel*ch)
	for c := range nChunks {
		for b := range nMel {
			copy(data[(c*nMel+b)*ch:(c*nMel+b+1)*ch], mel[b*frames+c*ch:b*frames+(c+1)*ch])
		}
	}
	x, err := tensor.New(data, []int{nChunks, 1, nMel, ch}, m.dev)
	if err != nil {
		return nil, err
	}
	for _, cv := range []struct{ w, b *tensor.Tensor }{{m.Conv1W, m.Conv1B}, {m.Conv2W, m.Conv2B}, {m.Conv3W, m.Conv3B}} {
		if x, err = ops.Conv2d(x, cv.w, cv.b, [2]int{2, 2}, [2]int{1, 1}); err != nil {
			return nil, err
		}
		if x, err = ops.GeluErf(x); err != nil {
			return nil, err
		}
	}
	// (nChunks, C, freq, time) → (nChunks, time, C*freq)
	C, freq, T := x.Dim(1), x.Dim(2), x.Dim(3)
	if T != cfg.ChunkPositions {
		return nil, fmt.Errorf("Qwen3ASR.Encode: %d post-conv frames per chunk, expected %d", T, cfg.ChunkPositions)
	}
	if x, err = autograd.Permute(x, 0, 3, 1, 2); err != nil {
		return nil, err
	}
	if x, err = autograd.Contiguous(x); err != nil {
		return nil, err
	}
	if x, err = autograd.Reshape(x, nChunks, T, C*freq); err != nil {
		return nil, err
	}
	if x, err = m.ConvOut.Forward(x); err != nil {
		return nil, err
	}
	if x, err = ops.Add(x, m.Pos); err != nil {
		return nil, err
	}
	// Valid post-conv positions form a prefix of the flattened chunks
	// (only the final chunk can be partial).
	S := cfg.AudioTokens(validFrames)
	D := cfg.DModel
	if x, err = autograd.Reshape(x, nChunks*T, D); err != nil {
		return nil, err
	}
	if x, err = x.Narrow(0, 0, S); err != nil {
		return nil, err
	}
	if x, err = autograd.Contiguous(x); err != nil {
		return nil, err
	}
	// Windowed attention: every n_window_infer mel frames of audio (8
	// chunks → 104 tokens) attend only among themselves. Pointwise layers
	// make each window an independent sequence, so full windows run as a
	// batch and the remainder as a second pass.
	W := cfg.ChunkPositions * (cfg.NWindowInfer / ch)
	nFull, rem := S/W, S%W
	var parts []*tensor.Tensor
	if nFull > 0 {
		h, err := x.Narrow(0, 0, nFull*W)
		if err != nil {
			return nil, err
		}
		if h, err = autograd.Reshape(h, nFull, W, D); err != nil {
			return nil, err
		}
		if h, err = m.encoderLayers(h); err != nil {
			return nil, err
		}
		if h, err = autograd.Reshape(h, nFull*W, D); err != nil {
			return nil, err
		}
		parts = append(parts, h)
	}
	if rem > 0 {
		h, err := x.Narrow(0, nFull*W, rem)
		if err != nil {
			return nil, err
		}
		if h, err = autograd.Contiguous(h); err != nil {
			return nil, err
		}
		if h, err = autograd.Reshape(h, 1, rem, D); err != nil {
			return nil, err
		}
		if h, err = m.encoderLayers(h); err != nil {
			return nil, err
		}
		if h, err = autograd.Reshape(h, rem, D); err != nil {
			return nil, err
		}
		parts = append(parts, h)
	}
	out := parts[0]
	if len(parts) > 1 {
		if out, err = ops.Concat(parts, 0); err != nil {
			return nil, err
		}
	}
	return m.LNPost.Forward(out)
}

// encoderLayers runs the pre-LN transformer stack on (B, S, D).
func (m *Qwen3ASR) encoderLayers(x *tensor.Tensor) (*tensor.Tensor, error) {
	B, S := x.Dim(0), x.Dim(1)
	for _, l := range m.Layers {
		h, err := l.LN1.Forward(x)
		if err != nil {
			return nil, err
		}
		if h, err = m.attend(l.Attn, h, B, S); err != nil {
			return nil, err
		}
		if x, err = ops.Add(x, h); err != nil {
			return nil, err
		}
		if h, err = l.LN2.Forward(x); err != nil {
			return nil, err
		}
		if h, err = l.FC1.Forward(h); err != nil {
			return nil, err
		}
		if h, err = ops.GeluErf(h); err != nil {
			return nil, err
		}
		if h, err = l.FC2.Forward(h); err != nil {
			return nil, err
		}
		if x, err = ops.Add(x, h); err != nil {
			return nil, err
		}
	}
	return x, nil
}

// attend is full bidirectional multi-head attention on (B, S, D).
func (m *Qwen3ASR) attend(a *qwen3ASRAttn, x *tensor.Tensor, B, S int) (*tensor.Tensor, error) {
	D, NH := m.Cfg.DModel, m.Cfg.Heads
	HD := D / NH
	q, err := a.Q.Forward(x)
	if err != nil {
		return nil, err
	}
	k, err := a.K.Forward(x)
	if err != nil {
		return nil, err
	}
	v, err := a.V.Forward(x)
	if err != nil {
		return nil, err
	}
	if q, err = headsForward(q, B, S, NH, HD); err != nil {
		return nil, err
	}
	if k, err = headsForward(k, B, S, NH, HD); err != nil {
		return nil, err
	}
	if v, err = headsForward(v, B, S, NH, HD); err != nil {
		return nil, err
	}
	if q, err = autograd.Reshape(q, B*NH, S, HD); err != nil {
		return nil, err
	}
	if k, err = autograd.Reshape(k, B*NH, S, HD); err != nil {
		return nil, err
	}
	if v, err = autograd.Reshape(v, B*NH, S, HD); err != nil {
		return nil, err
	}
	kT, err := autograd.Transpose(k, 1, 2)
	if err != nil {
		return nil, err
	}
	if kT, err = autograd.Contiguous(kT); err != nil {
		return nil, err
	}
	s, err := autograd.MatMul(q, kT)
	if err != nil {
		return nil, err
	}
	if s, err = ops.Mul(s, m.scale); err != nil {
		return nil, err
	}
	if s, err = ops.Softmax(s, -1); err != nil {
		return nil, err
	}
	ctx, err := autograd.MatMul(s, v)
	if err != nil {
		return nil, err
	}
	if ctx, err = autograd.Reshape(ctx, B, NH, S, HD); err != nil {
		return nil, err
	}
	if ctx, err = autograd.Permute(ctx, 0, 2, 1, 3); err != nil {
		return nil, err
	}
	if ctx, err = autograd.Contiguous(ctx); err != nil {
		return nil, err
	}
	if ctx, err = autograd.Reshape(ctx, B, S, D); err != nil {
		return nil, err
	}
	return a.O.Forward(ctx)
}

// Project maps encoder output (S, DModel) to decoder embeddings
// (S, OutputDim) in the decoder's activation dtype.
func (m *Qwen3ASR) Project(enc *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := m.Proj1.Forward(enc)
	if err != nil {
		return nil, err
	}
	if h, err = ops.GeluErf(h); err != nil {
		return nil, err
	}
	if h, err = m.Proj2.Forward(h); err != nil {
		return nil, err
	}
	if m.LM.Cfg.WeightDType == dtype.BF16 {
		return ops.Cast(h, dtype.BF16)
	}
	return h, nil
}

// PromptIDs builds the chat prompt the processor produces for one audio
// clip with nAudio placeholder tokens:
//
//	<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n<|audio_start|>
//	<|audio_pad|>×n<|audio_end|><|im_end|>\n<|im_start|>assistant\n{prefill}
//
// system and prefill are pre-tokenized (the system prompt carries hotwords
// or context; prefill "language English<asr_text>" forces the language).
// The returned offset is the index of the first audio token.
func (m *Qwen3ASR) PromptIDs(nAudio int, system, prefill []int64) (ids []int64, audioOffset int) {
	const imStart, imEnd, nl, sys, user, assistant = 151644, 151645, 198, 8948, 872, 77091
	ids = append(ids, imStart, sys, nl)
	ids = append(ids, system...)
	ids = append(ids, imEnd, nl, imStart, user, nl, m.Cfg.AudioStartID)
	audioOffset = len(ids)
	for range nAudio {
		ids = append(ids, m.Cfg.AudioTokenID)
	}
	ids = append(ids, m.Cfg.AudioEndID, imEnd, nl, imStart, assistant, nl)
	return append(ids, prefill...), audioOffset
}

// Prefill embeds the prompt, splices the projected audio (S, D) in place
// of the placeholder run starting at audioOffset, runs the decoder once
// and returns the last-position logits (V) on the CPU. cache receives
// the prompt's K/V.
func (m *Qwen3ASR) Prefill(ids []int64, audioOffset int, audio *tensor.Tensor, cache *KVCache) ([]float32, error) {
	S := audio.Dim(0)
	if audioOffset+S > len(ids) {
		return nil, fmt.Errorf("Qwen3ASR.Prefill: %d audio rows do not fit at offset %d of %d ids", S, audioOffset, len(ids))
	}
	tok, err := tensor.New(ids, []int{1, len(ids)}, m.dev)
	if err != nil {
		return nil, err
	}
	emb, err := m.LM.Embed(tok) // (1, T, D)
	if err != nil {
		return nil, err
	}
	D := emb.Dim(2)
	a3, err := autograd.Reshape(audio, 1, S, D)
	if err != nil {
		return nil, err
	}
	parts := []*tensor.Tensor{}
	if audioOffset > 0 {
		pre, err := emb.Narrow(1, 0, audioOffset)
		if err != nil {
			return nil, err
		}
		parts = append(parts, pre)
	}
	parts = append(parts, a3)
	if tail := len(ids) - audioOffset - S; tail > 0 {
		post, err := emb.Narrow(1, audioOffset+S, tail)
		if err != nil {
			return nil, err
		}
		parts = append(parts, post)
	}
	if emb, err = ops.Concat(parts, 1); err != nil {
		return nil, err
	}
	logits, err := m.LM.ForwardEmbeds(emb, cache)
	if err != nil {
		return nil, err
	}
	return lastRow(logits)
}

// lastRow copies the final (1, T, V) logit row to the host.
func lastRow(logits *tensor.Tensor) ([]float32, error) {
	T, V := logits.Dim(1), logits.Dim(2)
	last, err := logits.Narrow(1, T-1, 1)
	if err != nil {
		return nil, err
	}
	if last, err = autograd.Contiguous(last); err != nil {
		return nil, err
	}
	cpu, err := last.To(device.CPU())
	if err != nil {
		return nil, err
	}
	return tensor.MustSlice[float32](cpu)[:V], nil
}

// Transcribe is the full greedy pipeline: encode mel, project, prefill the
// prompt and decode up to maxTokens ids (EOS excluded). The returned ids
// start with the model's own "language <Name><asr_text>" prefix unless
// prefill already supplied it.
func (m *Qwen3ASR) Transcribe(mel []float32, validFrames int, system, prefill []int64, maxTokens int) ([]int64, error) {
	enc, err := m.Encode(mel, validFrames)
	if err != nil {
		return nil, err
	}
	audio, err := m.Project(enc)
	if err != nil {
		return nil, err
	}
	return m.TranscribeAudio(audio, system, prefill, maxTokens)
}

// TranscribeAudio is Transcribe on already projected audio embeddings.
func (m *Qwen3ASR) TranscribeAudio(audio *tensor.Tensor, system, prefill []int64, maxTokens int) ([]int64, error) {
	if maxTokens <= 0 {
		maxTokens = 512
	}
	ids, off := m.PromptIDs(audio.Dim(0), system, prefill)
	cache := NewKVCache(m.LM.Cfg.NumLayers)
	row, err := m.Prefill(ids, off, audio, cache)
	if err != nil {
		return nil, err
	}
	var out []int64
	for range maxTokens {
		best := int64(argmax(row))
		if m.isEOS(best) {
			break
		}
		out = append(out, best)
		tok, err := tensor.New([]int64{best}, []int{1, 1}, m.dev)
		if err != nil {
			return out, err
		}
		logits, err := m.LM.ForwardWithCache(tok, cache)
		if err != nil {
			return out, err
		}
		if row, err = lastRow(logits); err != nil {
			return out, err
		}
	}
	return out, nil
}

func (m *Qwen3ASR) isEOS(id int64) bool {
	for _, e := range m.Cfg.EOS {
		if id == e {
			return true
		}
	}
	return false
}
