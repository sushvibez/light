package engine

import (
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"light/gguf"
	"light/hub"
)

// Entry is one chat model the catalog found on disk.
type Entry struct {
	Spec      Spec   `json:"spec"`
	Kind      string `json:"kind"`       // "safetensors" or "gguf"
	ModelType string `json:"model_type"` // llama, mistral, qwen2, phi3, ...
	Params    string `json:"params"`     // e.g. "0.5B" (from the name when known)
	SizeBytes int64  `json:"size_bytes"` // weight bytes on disk
	VocabSize int    `json:"vocab_size"`
}

var supportedTypes = map[string]bool{"llama": true, "mistral": true, "qwen2": true, "qwen2_5": true, "qwen3": true, "qwen3_moe": true, "qwen3moe": true, "mellum": true, "qwen3_5": true, "qwen3_5_text": true, "muse_glimmer": true, "muse_glimmer_text": true, "sdar": true, "phi3": true, "gemma3_text": true, "gemma3": true, "gemma2": true, "gemma4": true, "gemma4_text": true, "falcon": true}

// Catalog scans the HF hub cache for supported causal LMs and the GGUF
// directory (LIGHT_GGUF_DIR or ~/.cache/light/gguf) for quantized files.
// GGUF entries use the tokenizer embedded in the file unless a sidecar
// `<file>.tokenizer.json` or a cached HF repo of the same family overrides
// it (those also carry generation_config EOS ids); they borrow the tokenizer from a sidecar `<file>.tokenizer.json`
// or, failing that, from the cached HF repo whose vocabulary size matches.
func Catalog() ([]Entry, error) {
	var out []Entry
	hf, _ := scanHFCache()
	out = append(out, hf...)
	gg, _ := scanGGUF(hf)
	out = append(out, gg...)
	sort.Slice(out, func(i, j int) bool { return out[i].Spec.ID < out[j].Spec.ID })
	return out, nil
}

type hfConfig struct {
	ModelType  string `json:"model_type"`
	VocabSize  int    `json:"vocab_size"`
	Dtype      string `json:"torch_dtype"`
	TextConfig *struct {
		VocabSize int `json:"vocab_size"`
	} `json:"text_config"` // multimodal wrappers (qwen3_asr)
}

// sttTypes are the speech-to-text model_types the Transcriber loads.
var sttTypes = map[string]bool{"whisper": true, "qwen3_asr": true}

func scanHFCache() ([]Entry, error) {
	root, err := hub.CacheDir()
	if err != nil {
		return nil, err
	}
	dirs, err := os.ReadDir(root)
	if err != nil {
		return nil, err
	}
	var out []Entry
	for _, d := range dirs {
		if !d.IsDir() || !strings.HasPrefix(d.Name(), "models--") {
			continue
		}
		repo := strings.ReplaceAll(strings.TrimPrefix(d.Name(), "models--"), "--", "/")
		snaps, err := filepath.Glob(filepath.Join(root, d.Name(), "snapshots", "*", "config.json"))
		if err != nil || len(snaps) == 0 {
			if e, ok := speechEntry(root, d.Name(), repo); ok {
				out = append(out, e)
			}
			continue
		}
		raw, err := os.ReadFile(snaps[len(snaps)-1])
		if err != nil {
			continue
		}
		var cfg hfConfig
		if json.Unmarshal(raw, &cfg) != nil || (!supportedTypes[cfg.ModelType] && !sttTypes[cfg.ModelType] && !embedTypes[cfg.ModelType]) {
			continue
		}
		if cfg.VocabSize == 0 && cfg.TextConfig != nil {
			cfg.VocabSize = cfg.TextConfig.VocabSize
		}
		snapDir := filepath.Dir(snaps[len(snaps)-1])
		if _, err := os.Stat(filepath.Join(snapDir, "tokenizer.json")); err != nil {
			continue
		}
		if sttTypes[cfg.ModelType] {
			var size int64
			if fi, err := os.Stat(filepath.Join(snapDir, "model.safetensors")); err == nil {
				size = fi.Size()
			}
			dt, params := "f32", whisperParams(repo)
			if cfg.ModelType == "qwen3_asr" {
				dt, params = "bf16", paramsFromName(repo)
			}
			out = append(out, Entry{
				Spec: Spec{ID: repo, Repo: repo, DType: dt, Device: "gpu"}, Kind: "stt",
				ModelType: cfg.ModelType, Params: params, SizeBytes: size, VocabSize: cfg.VocabSize,
			})
			continue
		}
		if embedTypes[cfg.ModelType] {
			var size int64
			if fi, err := os.Stat(filepath.Join(snapDir, "model.safetensors")); err == nil {
				size = fi.Size()
			}
			out = append(out, Entry{
				Spec: Spec{ID: repo, Repo: repo, DType: "bf16", Device: "gpu"}, Kind: "embedding",
				ModelType: cfg.ModelType, Params: "270M", SizeBytes: size, VocabSize: cfg.VocabSize,
			})
			continue
		}
		var size int64
		kind := "safetensors"
		sts, _ := filepath.Glob(filepath.Join(snapDir, "*.safetensors"))
		if len(sts) == 0 { // torch zip checkpoints (falcon-rw-1b) load through light/pth
			sts, _ = filepath.Glob(filepath.Join(snapDir, "pytorch_model*.bin"))
			kind = "pytorch"
		}
		if len(sts) == 0 {
			continue
		}
		for _, p := range sts {
			if fi, err := os.Stat(p); err == nil {
				size += fi.Size()
			}
		}
		dt := "bf16"
		if size > 6<<30 { // >6 GB of weights: BF16 is the only option on a 16 GB card
			dt = "bf16"
		} else if cfg.Dtype == "float32" {
			dt = "f32"
		}
		out = append(out, Entry{
			Spec:      Spec{ID: repo, Repo: repo, DType: dt, Device: "gpu"},
			Kind:      kind,
			ModelType: cfg.ModelType,
			Params:    paramsFromName(repo),
			SizeBytes: size,
			VocabSize: cfg.VocabSize,
		})
	}
	return out, nil
}

// speechEntry recognises a Moshi repo (no config.json; a moshi-core GGUF
// with the depformer tensors, e.g. kyutai/moshiko-candle-q8) and lists it
// as kind "speech".
func speechEntry(root, dir, repo string) (Entry, bool) {
	ggufs, _ := filepath.Glob(filepath.Join(root, dir, "snapshots", "*", "*.gguf"))
	for _, p := range ggufs {
		fh, err := os.Open(p)
		if err != nil {
			continue
		}
		g, err := gguf.Open(fh)
		fh.Close()
		if err != nil {
			continue
		}
		if _, ok := g.Tensors["depformer.0.linear_in.weight"]; !ok {
			continue
		}
		var size int64
		if fi, err := os.Stat(p); err == nil {
			size = fi.Size()
		}
		dt := "q8_0"
		if info, ok := g.Tensors["text_linear.weight"]; ok {
			dt = strings.ToLower(info.DType.String())
		}
		return Entry{
			Spec: Spec{ID: repo, Repo: repo, GGUF: p, DType: dt, Device: "gpu"}, Kind: "speech",
			ModelType: "moshi", Params: "7.7B", SizeBytes: size, VocabSize: 32000,
		}, true
	}
	return Entry{}, false
}

func scanGGUF(hf []Entry) ([]Entry, error) {
	dir := ggufDir()
	files, err := filepath.Glob(filepath.Join(dir, "*.gguf"))
	if err != nil {
		return nil, err
	}
	var out []Entry
	for _, p := range files {
		fh, err := os.Open(p)
		if err != nil {
			continue
		}
		g, err := gguf.Open(fh)
		if err != nil {
			fh.Close()
			continue
		}
		arch, _ := g.String("general.architecture")
		tokModel, _ := g.String("tokenizer.ggml.model")
		vocab := 0
		if toks, ok := g.Meta["tokenizer.ggml.tokens"].([]any); ok {
			vocab = len(toks)
		}
		fh.Close()
		if !supportedTypes[arch] {
			continue
		}
		fi, _ := os.Stat(p)
		e := Entry{
			Spec:      Spec{ID: "gguf/" + filepath.Base(p), GGUF: p, Device: "gpu"},
			Kind:      "gguf",
			ModelType: arch,
			Params:    paramsFromName(p),
			VocabSize: vocab,
		}
		if fi != nil {
			e.SizeBytes = fi.Size()
		}
		if side := strings.TrimSuffix(p, ".gguf") + ".tokenizer.json"; fileExists(side) {
			e.Spec.TokenizerPath = side
		} else if repo := matchTokenizerRepo(hf, arch, vocab, filepath.Base(p)); repo != "" {
			e.Spec.Repo = repo
		} else if tokModel != "llama" && tokModel != "gpt2" {
			continue // no sidecar, no HF repo, and no embedded tokenizer we can read
		}
		// Otherwise engine.Load builds the tokenizer from tokenizer.ggml.*.
		out = append(out, e)
	}
	return out, nil
}

// matchTokenizerRepo finds a cached HF repo of the same family with the
// same vocabulary size; among several, the one whose name shares the most
// prefix with the GGUF file name wins.
func matchTokenizerRepo(hf []Entry, arch string, vocab int, base string) string {
	family := map[string]string{"qwen2_5": "qwen2", "qwen3_moe": "qwen3", "qwen3moe": "qwen3"}
	norm := func(s string) string {
		s = strings.ToLower(s)
		if f, ok := family[s]; ok {
			return f
		}
		return s
	}
	best, bestScore := "", -1
	b := strings.ToLower(base)
	for _, e := range hf {
		if norm(e.ModelType) != norm(arch) || e.VocabSize != vocab {
			continue
		}
		name := strings.ToLower(strings.ReplaceAll(filepath.Base(e.Spec.Repo), "-", ""))
		score := 0
		for _, part := range strings.FieldsFunc(b, func(r rune) bool { return r == '-' || r == '_' || r == '.' }) {
			if strings.Contains(name, part) {
				score += len(part)
			}
		}
		if score > bestScore {
			best, bestScore = e.Spec.Repo, score
		}
	}
	return best
}

var paramsRe = regexp.MustCompile(`(?i)(?:^|[-_/ ])(\d+(?:\.\d+)?)([bm])(?:[-_. ]|$)`)

// paramsFromName extracts "0.5B" / "7B" / "270M" from a model name.
func paramsFromName(name string) string {
	m := paramsRe.FindStringSubmatch(strings.ToLower(name))
	if m == nil {
		return ""
	}
	return m[1] + strings.ToUpper(m[2])
}

// whisperParams maps the OpenAI size names to parameter counts.
func whisperParams(repo string) string {
	n := strings.ToLower(repo)
	for _, p := range []struct{ k, v string }{{"large", "1.5B"}, {"medium", "769M"}, {"small", "244M"}, {"base", "74M"}, {"tiny", "39M"}} {
		if strings.Contains(n, p.k) {
			return p.v
		}
	}
	return ""
}

func fileExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}
