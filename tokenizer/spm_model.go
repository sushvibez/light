package tokenizer

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"strings"
)

// SentencePieceModel is a SentencePiece `.model` file (the ModelProto
// protobuf) read without a protobuf runtime and turned into a Unigram
// tokenizer: the pieces with their scores and types plus the trainer and
// normalizer flags that matter at inference. The HF `tokenizer.json` path
// covers checkpoints that ship a converted tokenizer; this one handles the
// raw model (Moshi's tokenizer_spm_32k_3.model and friends).
//
// Normalization support is the identity rule (empty precompiled charsmap)
// plus the dummy-prefix / whitespace-escape / extra-whitespace options; an
// NFKC charsmap is reported in HasCharsMap and left unapplied.
type SentencePieceModel struct {
	*Unigram
	Pieces []string
	Scores []float32
	Types  []int32 // 1 normal, 2 unknown, 3 control, 4 user-defined, 5 unused, 6 byte

	ModelType              int32 // 1 unigram, 2 bpe, 3 word, 4 char
	AddDummyPrefix         bool
	RemoveExtraWhitespaces bool
	EscapeWhitespaces      bool
	ByteFallback           bool
	Normalizer             string // normalizer_spec.name ("identity", "nmt_nfkc", …)
	HasCharsMap            bool   // a precompiled charsmap is present (unapplied)
	UnkID, BosID, EosID    int32
	PadID                  int32
}

// SentencePiece piece types (sentencepiece_model.proto).
const (
	spmTypeNormal  = 1
	spmTypeUnknown = 2
	spmTypeControl = 3
	spmTypeUser    = 4
	spmTypeUnused  = 5
	spmTypeByte    = 6
)

// LoadSentencePieceModel reads a `.model` file.
func LoadSentencePieceModel(path string) (*SentencePieceModel, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	m, err := ParseSentencePieceModel(data)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return m, nil
}

// ParseSentencePieceModel parses the ModelProto bytes. Only unigram models
// are accepted: the Viterbi segmentation is what sentencepiece runs for
// them, while a BPE `.model` needs its merge ranks (use the tokenizer.json
// path for those).
func ParseSentencePieceModel(data []byte) (*SentencePieceModel, error) {
	m := &SentencePieceModel{AddDummyPrefix: true, RemoveExtraWhitespaces: true, EscapeWhitespaces: true, UnkID: 0, BosID: 1, EosID: 2, PadID: -1}
	err := pbWalk(data, func(field int, v uint64, b []byte) error {
		switch field {
		case 1: // repeated SentencePiece pieces
			piece, score, typ := "", float32(0), int32(spmTypeNormal)
			if err := pbWalk(b, func(f int, v uint64, bb []byte) error {
				switch f {
				case 1:
					piece = string(bb)
				case 2:
					score = math.Float32frombits(uint32(v))
				case 3:
					typ = int32(v)
				}
				return nil
			}); err != nil {
				return err
			}
			m.Pieces = append(m.Pieces, piece)
			m.Scores = append(m.Scores, score)
			m.Types = append(m.Types, typ)
		case 2: // TrainerSpec
			return pbWalk(b, func(f int, v uint64, bb []byte) error {
				switch f {
				case 3:
					m.ModelType = int32(v)
				case 35:
					m.ByteFallback = v != 0
				case 40:
					m.UnkID = int32(v)
				case 41:
					m.BosID = int32(v)
				case 42:
					m.EosID = int32(v)
				case 43:
					m.PadID = int32(v)
				}
				return nil
			})
		case 3: // NormalizerSpec
			return pbWalk(b, func(f int, v uint64, bb []byte) error {
				switch f {
				case 1:
					m.Normalizer = string(bb)
				case 2:
					m.HasCharsMap = len(bb) > 0
				case 3:
					m.AddDummyPrefix = v != 0
				case 4:
					m.RemoveExtraWhitespaces = v != 0
				case 5:
					m.EscapeWhitespaces = v != 0
				}
				return nil
			})
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	if len(m.Pieces) == 0 {
		return nil, fmt.Errorf("sentencepiece: no pieces found")
	}
	if m.ModelType != 1 {
		return nil, fmt.Errorf("sentencepiece: model_type %d not supported (unigram only)", m.ModelType)
	}
	vocab := make([]UnigramPiece, len(m.Pieces))
	for i, p := range m.Pieces {
		vocab[i] = UnigramPiece{Piece: p, Score: float64(m.Scores[i])}
		if m.Types[i] == spmTypeUnknown {
			m.UnkID = int32(i)
		}
	}
	cfg := UnigramConfig{UnkID: m.UnkID, ByteFallback: m.ByteFallback, Replacement: '▁', PrependScheme: "never", Split: true}
	if m.AddDummyPrefix {
		cfg.PrependScheme = "always"
	}
	if m.Unigram, err = NewUnigram(vocab, cfg); err != nil {
		return nil, err
	}
	return m, nil
}

// pbWalk iterates a protobuf message, calling fn with the field number, the
// scalar value (varint / fixed) and the bytes of a length-delimited field.
func pbWalk(b []byte, fn func(field int, v uint64, bytes []byte) error) error {
	for len(b) > 0 {
		key, n := binary.Uvarint(b)
		if n <= 0 {
			return fmt.Errorf("sentencepiece: bad protobuf key")
		}
		b = b[n:]
		field, wire := int(key>>3), int(key&7)
		switch wire {
		case 0:
			v, n := binary.Uvarint(b)
			if n <= 0 {
				return fmt.Errorf("sentencepiece: bad varint in field %d", field)
			}
			b = b[n:]
			if err := fn(field, v, nil); err != nil {
				return err
			}
		case 1:
			if len(b) < 8 {
				return fmt.Errorf("sentencepiece: truncated fixed64 in field %d", field)
			}
			if err := fn(field, binary.LittleEndian.Uint64(b), nil); err != nil {
				return err
			}
			b = b[8:]
		case 2:
			l, n := binary.Uvarint(b)
			if n <= 0 || uint64(len(b)-n) < l {
				return fmt.Errorf("sentencepiece: truncated bytes in field %d", field)
			}
			b = b[n:]
			if err := fn(field, 0, b[:l]); err != nil {
				return err
			}
			b = b[l:]
		case 5:
			if len(b) < 4 {
				return fmt.Errorf("sentencepiece: truncated fixed32 in field %d", field)
			}
			if err := fn(field, uint64(binary.LittleEndian.Uint32(b)), nil); err != nil {
				return err
			}
			b = b[4:]
		default:
			return fmt.Errorf("sentencepiece: unsupported wire type %d in field %d", wire, field)
		}
	}
	return nil
}

// Encode tokenizes text the way sentencepiece does for a unigram model:
// whitespace cleanup when the model asks for it, dummy prefix, ▁ escape,
// Viterbi segmentation, byte fallback for uncovered characters.
func (m *SentencePieceModel) Encode(text string) []int32 {
	if m.RemoveExtraWhitespaces {
		text = strings.Join(strings.Fields(text), " ")
	}
	// sentencepiece prepends the dummy space unconditionally; the Metaspace
	// pre-tokenizer only adds one when the text does not already start with
	// ▁, so leading spaces are handled here.
	if m.AddDummyPrefix && text != "" {
		text = " " + text
	}
	return m.Unigram.Encode(text)
}

// Decode turns ids back into text: control, unknown and unused pieces
// vanish, byte pieces become raw bytes, ▁ becomes a space and the dummy
// prefix is dropped. Invalid UTF-8 left by partial byte sequences is
// replaced by U+FFFD, as the reference decoder does.
func (m *SentencePieceModel) Decode(ids []int32) string {
	keep := make([]int32, 0, len(ids))
	for _, id := range ids {
		if id < 0 || int(id) >= len(m.Pieces) {
			continue
		}
		switch m.Types[id] {
		case spmTypeControl, spmTypeUnknown, spmTypeUnused:
			continue
		}
		keep = append(keep, id)
	}
	return strings.ToValidUTF8(m.Unigram.Decode(keep), "�")
}

// IDToPiece returns the surface form of id ("" when out of range).
func (m *SentencePieceModel) IDToPiece(id int32) string {
	if id < 0 || int(id) >= len(m.Pieces) {
		return ""
	}
	return m.Pieces[id]
}
