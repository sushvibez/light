package models

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// RefJSON is a transformers reference written by scripts/verify/ref_json.py,
// typically on Google Colab for checkpoints the local card cannot hold in
// BF16. Light verifies a quantized GGUF of the same checkpoint against it
// (models/heavy_ref_test.go): the prompt arrives pre-tokenized, so the
// comparison needs no tokenizer, and the tolerance scales with the quant.
type RefJSON struct {
	Repo         string   `json:"repo"`
	ModelType    string   `json:"model_type"`
	VocabSize    int      `json:"vocab_size"`
	DType        string   `json:"dtype"`
	Prompt       string   `json:"prompt"`
	ChatTemplate bool     `json:"chat_template"`
	IDs          []int64  `json:"ids"`
	Top          []RefTop `json:"top"`
	Greedy       []int64  `json:"greedy"`
	GreedyText   string   `json:"greedy_text"`
	GGUF         string   `json:"gguf"`      // glob of the local GGUF file name(s) to verify
	Tolerance    float64  `json:"tolerance"` // |logit − ref| allowed; 0 = by quant level
	Source       struct {
		Transformers string `json:"transformers"`
		Device       string `json:"device"`
	} `json:"source"`
	Path string `json:"-"`
}

// RefTop is one of the reference's top-k next tokens.
type RefTop struct {
	ID    int     `json:"id"`
	Logit float32 `json:"logit"`
	Text  string  `json:"text"`
}

// LoadRefJSON reads one reference file.
func LoadRefJSON(path string) (*RefJSON, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var r RefJSON
	if err := json.Unmarshal(raw, &r); err != nil {
		return nil, err
	}
	r.Path = path
	return &r, nil
}

// LoadRefJSONs reads every *.json under dir (models/testdata/refs).
func LoadRefJSONs(dir string) ([]*RefJSON, error) {
	paths, err := filepath.Glob(filepath.Join(dir, "*.json"))
	if err != nil {
		return nil, err
	}
	var out []*RefJSON
	for _, p := range paths {
		r, err := LoadRefJSON(p)
		if err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, nil
}
