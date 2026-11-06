package tokenizer

import (
	"encoding/json"
	"fmt"
)

// PostProcessor reproduces HF's `TemplateProcessing` / `BertProcessing`
// post-processors: it wraps one or two encoded sequences with the special
// tokens a model expects (for BERT: `[CLS] A [SEP]` and
// `[CLS] A [SEP] B [SEP]`).
type PostProcessor struct {
	single []templatePiece
	pair   []templatePiece
}

// templatePiece is one slot of a template: either a literal special-token id
// or a reference to input sequence "A" or "B".
type templatePiece struct {
	special bool
	id      int32
	seq     string
}

// NewBertPostProcessor builds the classic BERT template from the [CLS] and
// [SEP] ids.
func NewBertPostProcessor(clsID, sepID int32) *PostProcessor {
	cls := templatePiece{special: true, id: clsID}
	sep := templatePiece{special: true, id: sepID}
	return &PostProcessor{
		single: []templatePiece{cls, {seq: "A"}, sep},
		pair:   []templatePiece{cls, {seq: "A"}, sep, {seq: "B"}, sep},
	}
}

// Apply fills the single-sequence template with ids.
func (p *PostProcessor) Apply(ids []int32) []int32 {
	if p == nil {
		return ids
	}
	return p.fill(p.single, ids, nil)
}

// ApplyPair fills the pair template with a and b. Falls back to the single
// template over a followed by b when no pair template is configured.
func (p *PostProcessor) ApplyPair(a, b []int32) []int32 {
	if p == nil {
		return append(append([]int32{}, a...), b...)
	}
	if len(p.pair) == 0 {
		return p.fill(p.single, append(append([]int32{}, a...), b...), nil)
	}
	return p.fill(p.pair, a, b)
}

func (p *PostProcessor) fill(tpl []templatePiece, a, b []int32) []int32 {
	out := make([]int32, 0, len(a)+len(b)+len(tpl))
	for _, piece := range tpl {
		switch {
		case piece.special:
			out = append(out, piece.id)
		case piece.seq == "B":
			out = append(out, b...)
		default:
			out = append(out, a...)
		}
	}
	return out
}

// ---- tokenizer.json parsing ----

type postProcessorJSON struct {
	Type          string                `json:"type"`
	Single        []templateEntryJSON   `json:"single"`
	Pair          []templateEntryJSON   `json:"pair"`
	SpecialTokens map[string]specialDef `json:"special_tokens"`
	// BertProcessing / RobertaProcessing: ["[SEP]", 102]
	Sep []json.RawMessage `json:"sep"`
	Cls []json.RawMessage `json:"cls"`
	// Sequence of processors
	Processors []json.RawMessage `json:"processors"`
}

type templateEntryJSON struct {
	SpecialToken *struct {
		ID string `json:"id"`
	} `json:"SpecialToken"`
	Sequence *struct {
		ID string `json:"id"`
	} `json:"Sequence"`
}

type specialDef struct {
	IDs []int32 `json:"ids"`
}

// compilePostProcessor parses a `post_processor` block. It returns nil (no
// error) when the block is absent or of a type that adds no tokens
// (ByteLevel), so callers can treat nil as "identity".
func compilePostProcessor(raw json.RawMessage) (*PostProcessor, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var cfg postProcessorJSON
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, fmt.Errorf("tokenizer: post_processor: %w", err)
	}
	switch cfg.Type {
	case "TemplateProcessing":
		single, err := compileTemplate(cfg.Single, cfg.SpecialTokens)
		if err != nil {
			return nil, err
		}
		pair, err := compileTemplate(cfg.Pair, cfg.SpecialTokens)
		if err != nil {
			return nil, err
		}
		return &PostProcessor{single: single, pair: pair}, nil
	case "BertProcessing", "RobertaProcessing":
		cls, err := processingPairID(cfg.Cls)
		if err != nil {
			return nil, fmt.Errorf("tokenizer: post_processor cls: %w", err)
		}
		sep, err := processingPairID(cfg.Sep)
		if err != nil {
			return nil, fmt.Errorf("tokenizer: post_processor sep: %w", err)
		}
		return NewBertPostProcessor(cls, sep), nil
	case "Sequence":
		// Use the first processor in the chain that produces a template.
		for _, sub := range cfg.Processors {
			p, err := compilePostProcessor(sub)
			if err != nil {
				return nil, err
			}
			if p != nil {
				return p, nil
			}
		}
		return nil, nil
	default:
		return nil, nil
	}
}

func compileTemplate(entries []templateEntryJSON, specials map[string]specialDef) ([]templatePiece, error) {
	var out []templatePiece
	for _, e := range entries {
		switch {
		case e.SpecialToken != nil:
			def, ok := specials[e.SpecialToken.ID]
			if !ok || len(def.IDs) == 0 {
				return nil, fmt.Errorf("tokenizer: post_processor references unknown special token %q", e.SpecialToken.ID)
			}
			for _, id := range def.IDs {
				out = append(out, templatePiece{special: true, id: id})
			}
		case e.Sequence != nil:
			out = append(out, templatePiece{seq: e.Sequence.ID})
		}
	}
	return out, nil
}

// processingPairID extracts the id from a ["[SEP]", 102] pair.
func processingPairID(pair []json.RawMessage) (int32, error) {
	if len(pair) != 2 {
		return 0, fmt.Errorf("expected [token, id] pair, got %d elements", len(pair))
	}
	var id int32
	if err := json.Unmarshal(pair[1], &id); err != nil {
		return 0, err
	}
	return id, nil
}
