package models

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"light/device"
	"light/hub"
	"light/safetensors"
	"light/tensor"
)

// statefulLM adapts a model with its own decode-state type C (GLM5Cache,
// DeepseekV41Cache, KimiK31Cache) to CausalLM: the state rides in
// KVCache.State, created on the first forward and dropped by Reset.
type statefulLM[C any] struct {
	fwd                   func(*tensor.Tensor, C) (*tensor.Tensor, error)
	newCache              func() C
	layers, vocab, maxCtx int
	params                int
}

func (s *statefulLM[C]) ForwardWithCache(tokens *tensor.Tensor, kv *KVCache) (*tensor.Tensor, error) {
	if kv == nil {
		var none C
		return s.fwd(tokens, none)
	}
	c, ok := kv.State.(C)
	if !ok {
		if kv.State != nil {
			return nil, fmt.Errorf("models: cache holds %T state", kv.State)
		}
		c = s.newCache()
		kv.State = c
	}
	out, err := s.fwd(tokens, c)
	if err == nil {
		kv.Len += tokens.Dim(1)
		kv.posSynced = false
	}
	return out, err
}

func (s *statefulLM[C]) NumLayers() int     { return s.layers }
func (s *statefulLM[C]) VocabSize() int     { return s.vocab }
func (s *statefulLM[C]) MaxContext() int    { return s.maxCtx }
func (s *statefulLM[C]) SlidingWindow() int { return 0 }
func (s *statefulLM[C]) ParamCount() int    { return s.params }

// snapshotDir downloads repo's config.json and safetensors shards (and the
// tokenizer files when present) and returns the folder holding them, for
// the loaders that read a checkpoint directory. A local folder is used as is.
func snapshotDir(repo string) (string, error) {
	if st, err := os.Stat(repo); err == nil && st.IsDir() {
		return repo, nil
	}
	cfg, err := hub.Download(repo, "main", "config.json")
	if err != nil {
		return "", err
	}
	if _, err := hub.DownloadModelSafetensors(repo, "main"); err != nil {
		return "", err
	}
	return filepath.Dir(cfg), nil
}

// statefulMeta reads max_position_embeddings (text_config first) from dir's
// config.json and counts the checkpoint's stored elements (packed FP4
// counts as one per byte) for ParamCount.
func statefulMeta(dir string) (maxCtx, params int, err error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return 0, 0, err
	}
	var c struct {
		Max  int `json:"max_position_embeddings"`
		Text *struct {
			Max int `json:"max_position_embeddings"`
		} `json:"text_config"`
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		return 0, 0, err
	}
	maxCtx = c.Max
	if c.Text != nil && c.Text.Max > 0 {
		maxCtx = c.Text.Max
	}
	if maxCtx <= 0 {
		maxCtx = 4096
	}
	paths, _ := filepath.Glob(filepath.Join(dir, "*.safetensors"))
	if len(paths) > 0 {
		hf, _, err := safetensors.LoadFiles(paths, device.CPU())
		if err != nil {
			return 0, 0, err
		}
		for _, t := range hf {
			params += t.NumElements()
		}
	}
	return maxCtx, params, nil
}

// LoadGLM5ChatFromHF loads GLM-5.3 (glm5_next) as a CausalLM for the
// engine; the decode state is GLM5Cache.
func LoadGLM5ChatFromHF(repo string, dev device.Device) (CausalLM, error) {
	dir, err := snapshotDir(repo)
	if err != nil {
		return nil, err
	}
	m, err := LoadGLM5(dir, dev)
	if err != nil {
		return nil, err
	}
	maxCtx, params, err := statefulMeta(dir)
	if err != nil {
		return nil, err
	}
	return &statefulLM[*GLM5Cache]{fwd: m.ForwardWithCache, newCache: m.NewCache,
		layers: len(m.Layers), vocab: m.Cfg.VocabSize, maxCtx: maxCtx, params: params}, nil
}

// LoadDeepseekV41ChatFromHF loads DeepSeek-V4.1 (deepseek_v41) as a
// CausalLM for the engine; the decode state is DeepseekV41Cache.
func LoadDeepseekV41ChatFromHF(repo string, dev device.Device) (CausalLM, error) {
	dir, err := snapshotDir(repo)
	if err != nil {
		return nil, err
	}
	m, err := LoadDeepseekV41(dir, dev)
	if err != nil {
		return nil, err
	}
	maxCtx, params, err := statefulMeta(dir)
	if err != nil {
		return nil, err
	}
	return &statefulLM[*DeepseekV41Cache]{fwd: m.ForwardWithCache, newCache: m.NewCache,
		layers: len(m.Layers), vocab: m.Cfg.VocabSize, maxCtx: maxCtx, params: params}, nil
}

// LoadKimiK31ChatFromHF loads Kimi K3.1 (kimi_k31) as a CausalLM for the
// engine; the decode state is KimiK31Cache.
func LoadKimiK31ChatFromHF(repo string, dev device.Device) (CausalLM, error) {
	dir, err := snapshotDir(repo)
	if err != nil {
		return nil, err
	}
	m, err := LoadKimiK31(dir, dev)
	if err != nil {
		return nil, err
	}
	maxCtx, params, err := statefulMeta(dir)
	if err != nil {
		return nil, err
	}
	return &statefulLM[*KimiK31Cache]{fwd: m.ForwardWithCache, newCache: m.NewCache,
		layers: len(m.Layers), vocab: m.Cfg.VocabSize, maxCtx: maxCtx, params: params}, nil
}
