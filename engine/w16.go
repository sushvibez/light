package engine

import (
	"encoding/json"
	"os"
	"strings"

	"light/dtype"
	"light/hub"
)

// w16StorageDType picks the 16-bit weight storage for dtype "w16": the
// checkpoint's own 16-bit dtype, so narrowing is lossless (BF16 for
// bfloat16, F16 for float16). ok is false for an F32 checkpoint, which
// then loads at F32. Unknown or missing dtypes take BF16.
func w16StorageDType(repo string) (dt dtype.DType, ok bool) {
	switch checkpointDType(repo) {
	case "float32":
		return dtype.F32, false
	case "float16":
		return dtype.F16, true
	}
	return dtype.BF16, true
}

// checkpointDType is config.json's torch_dtype (or transformers 5's
// dtype), from the text config of a multimodal wrapper when the top level
// lacks it; "" when unknown.
func checkpointDType(repo string) string {
	path, err := hub.Download(repo, "main", "config.json")
	if err != nil {
		return ""
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	type dts struct {
		Torch string `json:"torch_dtype"`
		DType string `json:"dtype"`
	}
	var c struct {
		dts
		Text *dts `json:"text_config"`
	}
	if json.Unmarshal(raw, &c) != nil {
		return ""
	}
	for _, d := range []*dts{&c.dts, c.Text} {
		if d == nil {
			continue
		}
		for _, s := range []string{d.Torch, d.DType} {
			if s = strings.TrimPrefix(strings.ToLower(s), "torch."); s != "" {
				return s
			}
		}
	}
	return ""
}
