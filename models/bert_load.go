package models

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"

	"light/device"
	"light/hub"
	"light/tensor"
)

// bertHFConfig is the subset of HuggingFace's BERT config.json we read.
type bertHFConfig struct {
	VocabSize         int     `json:"vocab_size"`
	HiddenSize        int     `json:"hidden_size"`
	NumHiddenLayers   int     `json:"num_hidden_layers"`
	NumAttentionHeads int     `json:"num_attention_heads"`
	IntermediateSize  int     `json:"intermediate_size"`
	MaxPositionEmbed  int     `json:"max_position_embeddings"`
	TypeVocabSize     int     `json:"type_vocab_size"`
	LayerNormEps      float64 `json:"layer_norm_eps"`
	HiddenAct         string  `json:"hidden_act"`
	PadTokenID        int     `json:"pad_token_id"`
	ModelType         string  `json:"model_type"`
}

// fetchBertConfig downloads (or resolves from the HF cache) config.json.
func fetchBertConfig(repo string) (BertConfig, error) {
	path, err := hub.Download(repo, "main", "config.json")
	if err != nil {
		return BertConfig{}, fmt.Errorf("fetchBertConfig: %w", err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return BertConfig{}, err
	}
	var hf bertHFConfig
	if err := json.Unmarshal(raw, &hf); err != nil {
		return BertConfig{}, fmt.Errorf("fetchBertConfig: parse: %w", err)
	}
	if hf.ModelType != "" && hf.ModelType != "bert" {
		return BertConfig{}, fmt.Errorf("fetchBertConfig: model_type %q is not bert", hf.ModelType)
	}
	if hf.LayerNormEps == 0 {
		hf.LayerNormEps = 1e-12
	}
	if hf.HiddenAct == "" {
		hf.HiddenAct = "gelu"
	}
	if hf.TypeVocabSize == 0 {
		hf.TypeVocabSize = 2
	}
	return BertConfig{
		VocabSize: hf.VocabSize, HiddenSize: hf.HiddenSize,
		NumLayers: hf.NumHiddenLayers, NumHeads: hf.NumAttentionHeads,
		IntermediateSize: hf.IntermediateSize, MaxPosition: hf.MaxPositionEmbed,
		TypeVocabSize: hf.TypeVocabSize, LayerNormEps: hf.LayerNormEps,
		HiddenAct: hf.HiddenAct, PadTokenID: hf.PadTokenID,
	}, nil
}

// bertKeyPrefix finds the prefix in front of `embeddings.word_embeddings.weight`:
// "" for bare sentence-transformer exports (unsloth/bge-small-en-v1.5),
// "bert." for `BertModel`/`BertForMaskedLM` checkpoints, "roberta." etc.
func bertKeyPrefix(hf map[string]*tensor.Tensor) (string, error) {
	const anchor = "embeddings.word_embeddings.weight"
	for k := range hf {
		if strings.HasSuffix(k, anchor) {
			return strings.TrimSuffix(k, anchor), nil
		}
	}
	return "", fmt.Errorf("HF BERT: no tensor ending in %q", anchor)
}

// LoadBertFromHF loads a BERT encoder from an HF repo (config.json +
// model.safetensors). The pooler is attached when the checkpoint carries
// `pooler.dense.*`. The HF cache is consulted first, so a cached repo loads
// offline.
//
// Key mapping (optionally prefixed with `bert.`):
//
//	embeddings.{word,position,token_type}_embeddings.weight, embeddings.LayerNorm.{weight,bias}
//	encoder.layer.{i}.attention.self.{query,key,value}.{weight,bias}
//	encoder.layer.{i}.attention.output.dense.{weight,bias} / .LayerNorm.{weight,bias}
//	encoder.layer.{i}.intermediate.dense.{weight,bias}
//	encoder.layer.{i}.output.dense.{weight,bias} / .LayerNorm.{weight,bias}
//	pooler.dense.{weight,bias}                 (optional)
//
// HF `nn.Linear` weights are (out, in) like ours, so nothing is transposed.
// `embeddings.position_ids` is a buffer and is ignored.
func LoadBertFromHF(repo string, dev device.Device) (*BertModel, error) {
	cfg, err := fetchBertConfig(repo)
	if err != nil {
		return nil, err
	}
	hf, err := loadSafetensorsAuto(repo)
	if err != nil {
		return nil, fmt.Errorf("LoadBertFromHF: %w", err)
	}
	prefix, err := bertKeyPrefix(hf)
	if err != nil {
		return nil, err
	}
	_, hasPooler := hf[prefix+"pooler.dense.weight"]
	m, err := NewBertModel(cfg, hasPooler, dev)
	if err != nil {
		return nil, err
	}
	if err := mapHFToBert(m, hf, prefix); err != nil {
		return nil, fmt.Errorf("LoadBertFromHF: %w", err)
	}
	return m, nil
}

// LoadBertMLMFromHF loads a `BertForMaskedLM` checkpoint: the encoder plus
// `cls.predictions.transform.dense.*`, `cls.predictions.transform.LayerNorm.*`
// and `cls.predictions.bias` (the decoder weight is tied to the word
// embeddings).
func LoadBertMLMFromHF(repo string, dev device.Device) (*BertForMaskedLM, error) {
	cfg, err := fetchBertConfig(repo)
	if err != nil {
		return nil, err
	}
	hf, err := loadSafetensorsAuto(repo)
	if err != nil {
		return nil, fmt.Errorf("LoadBertMLMFromHF: %w", err)
	}
	prefix, err := bertKeyPrefix(hf)
	if err != nil {
		return nil, err
	}
	m, err := NewBertForMaskedLM(cfg, dev)
	if err != nil {
		return nil, err
	}
	if err := mapHFToBert(m.Bert, hf, prefix); err != nil {
		return nil, fmt.Errorf("LoadBertMLMFromHF: %w", err)
	}
	const cls = "cls.predictions."
	if err := copyExactCast(m.Transform.W, hf, cls+"transform.dense.weight"); err != nil {
		return nil, err
	}
	if err := copyExactCast(m.Transform.B, hf, cls+"transform.dense.bias"); err != nil {
		return nil, err
	}
	if err := copyExactCast(m.TransformLN.Gamma, hf, cls+"transform.LayerNorm.weight"); err != nil {
		return nil, err
	}
	if err := copyExactCast(m.TransformLN.Beta, hf, cls+"transform.LayerNorm.bias"); err != nil {
		return nil, err
	}
	biasKey := cls + "bias"
	if _, ok := hf[biasKey]; !ok {
		biasKey = cls + "decoder.bias"
	}
	if err := copyExactCast(m.Decoder.B, hf, biasKey); err != nil {
		return nil, err
	}
	return m, nil
}

func mapHFToBert(m *BertModel, hf map[string]*tensor.Tensor, prefix string) error {
	cp := func(dst *tensor.Tensor, name string) error {
		return copyExactCast(dst, hf, prefix+name)
	}
	if err := cp(m.WordEmb.Weight, "embeddings.word_embeddings.weight"); err != nil {
		return err
	}
	if err := cp(m.PosEmb.Weight, "embeddings.position_embeddings.weight"); err != nil {
		return err
	}
	if err := cp(m.TypeEmb.Weight, "embeddings.token_type_embeddings.weight"); err != nil {
		return err
	}
	if err := cp(m.EmbLN.Gamma, "embeddings.LayerNorm.weight"); err != nil {
		return err
	}
	if err := cp(m.EmbLN.Beta, "embeddings.LayerNorm.bias"); err != nil {
		return err
	}
	for i, l := range m.Layers {
		p := "encoder.layer." + strconv.Itoa(i) + "."
		pairs := []struct {
			dst  *tensor.Tensor
			name string
		}{
			{l.Q.W, p + "attention.self.query.weight"},
			{l.Q.B, p + "attention.self.query.bias"},
			{l.K.W, p + "attention.self.key.weight"},
			{l.K.B, p + "attention.self.key.bias"},
			{l.V.W, p + "attention.self.value.weight"},
			{l.V.B, p + "attention.self.value.bias"},
			{l.AttnOut.W, p + "attention.output.dense.weight"},
			{l.AttnOut.B, p + "attention.output.dense.bias"},
			{l.AttnLN.Gamma, p + "attention.output.LayerNorm.weight"},
			{l.AttnLN.Beta, p + "attention.output.LayerNorm.bias"},
			{l.Inter.W, p + "intermediate.dense.weight"},
			{l.Inter.B, p + "intermediate.dense.bias"},
			{l.Out.W, p + "output.dense.weight"},
			{l.Out.B, p + "output.dense.bias"},
			{l.OutLN.Gamma, p + "output.LayerNorm.weight"},
			{l.OutLN.Beta, p + "output.LayerNorm.bias"},
		}
		for _, pr := range pairs {
			if err := cp(pr.dst, pr.name); err != nil {
				return err
			}
		}
	}
	if m.Pooler != nil {
		if err := cp(m.Pooler.W, "pooler.dense.weight"); err != nil {
			return err
		}
		if err := cp(m.Pooler.B, "pooler.dense.bias"); err != nil {
			return err
		}
	}
	return nil
}
