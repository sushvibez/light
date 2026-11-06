package engine

import (
	"errors"
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"light/device"
	"light/dtype"
	"light/hub"
	"light/models"
	"light/tensor"
	"light/tokenizer"
)

// embedTypes are the HF model_types the Embedder loads.
var embedTypes = map[string]bool{"embedding_gemma2": true}

// Embedder is a loaded text-embedding checkpoint (EmbeddingGemma 2) with
// its tokenizer. Embed runs the sentence-transformers pipeline: tokenize
// with <bos> … <eos>, encode, mean-pool, L2-normalise.
type Embedder struct {
	Spec     Spec
	Model    *models.EmbeddingGemma2
	Tok      *tokenizer.BPE
	LoadTime time.Duration
	bos, eos int64
	mu       sync.Mutex
}

// LoadEmbedder loads an embedding repo (unsloth/embeddinggemma-2).
// Spec.DType picks the weight precision (bf16 default).
func LoadEmbedder(spec Spec) (*Embedder, error) {
	t0 := time.Now()
	dev, err := pickDevice(spec.Device)
	if err != nil {
		return nil, err
	}
	if spec.Repo == "" {
		return nil, errors.New("engine: embedder spec needs Repo")
	}
	mt, err := hfModelType(spec.Repo)
	if err != nil {
		return nil, err
	}
	if !embedTypes[mt] {
		return nil, fmt.Errorf("engine: model_type %q is not an embedding model", mt)
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
	wdt := dtype.BF16
	if strings.EqualFold(spec.DType, "f32") {
		wdt = dtype.F32
	}
	m, err := models.LoadEmbeddingGemma2FromHF(spec.Repo, dev, wdt)
	if err != nil {
		return nil, err
	}
	e := &Embedder{Spec: spec, Model: m, Tok: tok, bos: int64(tok.TokenToID("<bos>")), eos: int64(tok.TokenToID("<eos>"))}
	e.Spec.Family = "embedding_gemma2"
	e.LoadTime = time.Since(t0)
	return e, nil
}

// Dim is the embedding width (768 for EmbeddingGemma 2).
func (e *Embedder) Dim() int { return e.Model.EmbeddingDim() }

// Encode tokenizes one text with the checkpoint's <bos> … <eos> template,
// truncated to the model's context.
func (e *Embedder) Encode(text string) []int64 {
	ids := make([]int64, 0, 16)
	if e.bos >= 0 {
		ids = append(ids, e.bos)
	}
	for _, id := range e.Tok.Encode(text) {
		ids = append(ids, int64(id))
	}
	if max := e.Model.MaxContext() - 1; len(ids) > max {
		ids = ids[:max]
	}
	if e.eos >= 0 {
		ids = append(ids, e.eos)
	}
	return ids
}

// Embed returns one L2-normalised embedding per text and the total token
// count. Texts are run as one padded batch.
func (e *Embedder) Embed(texts []string) ([][]float32, int, error) {
	if len(texts) == 0 {
		return nil, 0, errors.New("engine: no input")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	ids := make([][]int64, len(texts))
	T, total := 0, 0
	for i, s := range texts {
		ids[i] = e.Encode(s)
		T = max(T, len(ids[i]))
		total += len(ids[i])
	}
	B := len(ids)
	flat, keep := make([]int64, B*T), make([]int64, B*T)
	for b, s := range ids {
		copy(flat[b*T:], s)
		for j := range s {
			keep[b*T+j] = 1
		}
	}
	dev := e.Model.TokenEmb.Weight.Device()
	tk, err := tensor.New(flat, []int{B, T}, dev)
	if err != nil {
		return nil, 0, err
	}
	var mk *tensor.Tensor
	if B > 1 {
		if mk, err = tensor.New(keep, []int{B, T}, dev); err != nil {
			return nil, 0, err
		}
	}
	emb, err := e.Model.Embed(tk, mk)
	if err != nil {
		return nil, 0, err
	}
	if emb, err = emb.To(device.CPU()); err != nil {
		return nil, 0, err
	}
	all := tensor.MustSlice[float32](emb)
	D := e.Dim()
	out := make([][]float32, B)
	for b := range out {
		out[b] = append([]float32(nil), all[b*D:(b+1)*D]...)
	}
	return out, total, nil
}
