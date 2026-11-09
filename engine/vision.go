package engine

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	_ "image/gif" // decoders for attached images
	_ "image/jpeg"
	_ "image/png"
	"os"
	"strings"

	"light/models"
	"light/ops"
	"light/tensor"
)

// UnmarshalJSON accepts content as a string or as OpenAI content parts:
// {"type": "text", "text": ...} and {"type": "image_url", "image_url":
// {"url": "data:image/...;base64,..."}} (inline images only; the server
// does not fetch URLs).
func (m *Message) UnmarshalJSON(b []byte) error {
	var raw struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(b, &raw); err != nil {
		return err
	}
	m.Role, m.Content, m.Images = raw.Role, "", nil
	if len(raw.Content) == 0 || string(raw.Content) == "null" {
		return nil
	}
	if raw.Content[0] == '"' {
		return json.Unmarshal(raw.Content, &m.Content)
	}
	var parts []struct {
		Type     string `json:"type"`
		Text     string `json:"text"`
		ImageURL *struct {
			URL string `json:"url"`
		} `json:"image_url"`
	}
	if err := json.Unmarshal(raw.Content, &parts); err != nil {
		return fmt.Errorf("message content: %w", err)
	}
	var text []string
	for _, p := range parts {
		switch p.Type {
		case "text":
			text = append(text, p.Text)
		case "image_url":
			if p.ImageURL == nil {
				return errors.New("image_url part without a url")
			}
			img, err := decodeDataURL(p.ImageURL.URL)
			if err != nil {
				return err
			}
			m.Images = append(m.Images, img)
		default:
			return fmt.Errorf("content part type %q not supported (text, image_url)", p.Type)
		}
	}
	m.Content = strings.Join(text, "\n")
	return nil
}

// MarshalJSON writes the text content (images are input-only).
func (m Message) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}{m.Role, m.Content})
}

func decodeDataURL(u string) (image.Image, error) {
	rest, ok := strings.CutPrefix(u, "data:")
	if !ok {
		return nil, errors.New("image_url: only data: URLs are accepted")
	}
	meta, data, ok := strings.Cut(rest, ",")
	if !ok || !strings.HasSuffix(meta, ";base64") {
		return nil, errors.New("image_url: want data:<type>;base64,<data>")
	}
	b, err := base64.StdEncoding.DecodeString(data)
	if err != nil {
		return nil, fmt.Errorf("image_url: %w", err)
	}
	img, _, err := image.Decode(bytes.NewReader(b))
	if err != nil {
		return nil, fmt.Errorf("image_url: %w", err)
	}
	return img, nil
}

// LoadImage decodes a PNG, JPEG or GIF file for Message.Images.
func LoadImage(path string) (image.Image, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return img, nil
}

func hasImages(msgs []Message) bool {
	for _, m := range msgs {
		if len(m.Images) > 0 {
			return true
		}
	}
	return false
}

// multimodal prepares an image + text prompt for a vision-language model:
// the placeholder token of each image expanded to its merged-patch count,
// the vision tower's tokens, and the three-stream positions. It returns a
// forward function for the generation loop (prefill, then decode steps at
// cache length + delta).
func (m *Model) multimodal(msgs []Message, ids []int32, cache *models.KVCache) (func([]int64) ([]float32, error), []int64, error) {
	lm, ok := m.LM.(*models.Qwen35)
	if !ok || lm.Cfg.ImageTokenID == 0 || len(lm.Cfg.MRoPESection) != 3 {
		return nil, nil, fmt.Errorf("engine: %s does not take images", m.Spec.ID)
	}
	if m.Template.Source != "jinja" {
		return nil, nil, errors.New("engine: images need the checkpoint's chat template")
	}
	if m.vision == nil {
		v, err := models.LoadQwen35VisionFromHF(m.Spec.Repo, m.Dev)
		if err != nil {
			return nil, nil, fmt.Errorf("engine: vision tower: %w", err)
		}
		m.vision = v
	}
	var imgs []image.Image
	for _, msg := range msgs {
		imgs = append(imgs, msg.Images...)
	}
	cfg := m.vision.ImageConfig()
	merge := cfg.MergeSize
	var grids [][3]int
	var feats []*tensor.Tensor
	for _, img := range imgs {
		px, grid, err := models.Qwen35ImagePatches(img, cfg)
		if err != nil {
			return nil, nil, err
		}
		pt, err := tensor.New(px, []int{grid[1] * grid[2], len(px) / (grid[1] * grid[2])}, m.Dev)
		if err != nil {
			return nil, nil, err
		}
		f, err := m.vision.Forward(pt, [][3]int{grid})
		if err != nil {
			return nil, nil, fmt.Errorf("engine: vision tower: %w", err)
		}
		grids, feats = append(grids, grid), append(feats, f)
	}
	// Expand each placeholder to its image's token count.
	tok := lm.Cfg.ImageTokenID
	var out []int64
	g := 0
	for _, id := range ids {
		if int64(id) != tok {
			out = append(out, int64(id))
			continue
		}
		if g >= len(grids) {
			return nil, nil, fmt.Errorf("engine: the prompt has more image placeholders than images (%d)", len(grids))
		}
		n := grids[g][0] * grids[g][1] * grids[g][2] / (merge * merge)
		for range n {
			out = append(out, tok)
		}
		g++
	}
	if g != len(grids) {
		return nil, nil, fmt.Errorf("engine: %d images but the template placed %d", len(grids), g)
	}
	pos, delta, err := models.Qwen35RopeIndex(out, tok, grids, merge)
	if err != nil {
		return nil, nil, err
	}
	img := feats[0]
	if len(feats) > 1 {
		parts := make([]*tensor.Tensor, len(feats))
		copy(parts, feats)
		if img, err = concatRows(parts); err != nil {
			return nil, nil, err
		}
	}
	first := true
	forward := func(tokens []int64) ([]float32, error) {
		x, err := tensor.New(tokens, []int{1, len(tokens)}, m.Dev)
		if err != nil {
			return nil, err
		}
		p, im := pos, img
		if !first {
			p0 := int64(cache.Len) + delta
			p, im = [3][]int64{}, nil
			for s := range 3 {
				for i := range tokens {
					p[s] = append(p[s], p0+int64(i))
				}
			}
		}
		first = false
		logits, err := lm.ForwardMultimodal(x, im, p, cache)
		if err != nil {
			return nil, err
		}
		return m.lastLogits(logits, len(tokens))
	}
	return forward, out, nil
}

func concatRows(parts []*tensor.Tensor) (*tensor.Tensor, error) { return ops.Concat(parts, 0) }
