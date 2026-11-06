package tokenizer

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// Reference values from the sentencepiece Python package on
// kyutai/moshiko-candle-q8 tokenizer_spm_32k_3.model (unigram, byte
// fallback, dummy prefix, identity normalizer).
func moshiSPMPath(t *testing.T) string {
	home, _ := os.UserHomeDir()
	m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--kyutai--moshiko-candle-q8/snapshots/*/tokenizer_spm_32k_3.model"))
	if len(m) == 0 {
		t.Skip("kyutai/moshiko-candle-q8 tokenizer not in HF cache")
	}
	return m[0]
}

func TestSentencePieceModelMoshi(t *testing.T) {
	m, err := LoadSentencePieceModel(moshiSPMPath(t))
	if err != nil {
		t.Fatal(err)
	}
	if m.VocabSize() != 32000 || m.ModelType != 1 || !m.ByteFallback || !m.AddDummyPrefix || m.RemoveExtraWhitespaces || m.HasCharsMap ||
		m.UnkID != 0 || m.BosID != 1 || m.EosID != 2 || m.PadID != 3 {
		t.Fatalf("model flags: %+v", *m)
	}
	for id, want := range map[int32]string{0: "<unk>", 3: "<pad>", 4: "<0x00>", 1000: "▁issue", 5000: "quad", 31999: "⊂"} {
		if got := m.IDToPiece(id); got != want {
			t.Errorf("piece %d = %q, want %q", id, got, want)
		}
	}
	if m.Scores[1000] != -9.039497375488281 || m.Scores[31999] != -15.095074653625488 {
		t.Errorf("scores %v %v", m.Scores[1000], m.Scores[31999])
	}
	enc := map[string][]int32{
		"Hello, how are you today?": {11725, 261, 452, 298, 304, 1497, 330},
		"I'm fine thanks! 123 ©":    {293, 286, 339, 1559, 2526, 430, 260, 265, 270, 278, 5446},
		"  double  space":           {260, 260, 1707, 260, 730},
		"naïve café ☃":              {572, 16650, 518, 29301, 260, 230, 156, 135},
	}
	for s, want := range enc {
		got := m.Encode(s)
		if !reflect.DeepEqual(got, want) {
			t.Errorf("Encode(%q) = %v, want %v", s, got, want)
		}
		if back := m.Decode(got); back != s {
			t.Errorf("Decode(Encode(%q)) = %q", s, back)
		}
	}
	dec := map[string][]int32{
		"\x01` issue": {5, 100, 1000},
		"issue issue": {1000, 3, 1000},
		"ޔ�":          {226, 152, 150},
	}
	for want, ids := range dec {
		if got := m.Decode(ids); got != want {
			t.Errorf("Decode(%v) = %q, want %q", ids, got, want)
		}
	}
}
