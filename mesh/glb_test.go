package mesh

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// The GLB parses back: header, chunk sizes, accessors and the binary
// data (positions, indices, clamped RGBA colours).
func TestWriteGLB(t *testing.T) {
	path := filepath.Join(t.TempDir(), "t.glb")
	m := tetra()
	if err := Write(path, m); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if os.Getenv("LIGHT_GLB_OUT") != "" {
		os.WriteFile(os.Getenv("LIGHT_GLB_OUT"), raw, 0o644)
	}
	le := binary.LittleEndian
	if le.Uint32(raw) != glbMagic || le.Uint32(raw[4:]) != 2 || int(le.Uint32(raw[8:])) != len(raw) {
		t.Fatalf("bad header % x", raw[:12])
	}
	jl := int(le.Uint32(raw[12:]))
	if le.Uint32(raw[16:]) != glbChunkJSON || jl%4 != 0 {
		t.Fatal("bad JSON chunk")
	}
	var doc struct {
		Accessors []struct {
			BufferView, ComponentType, Count int
			Type                             string
			Normalized                       bool
			Min, Max                         []float64
		}
		BufferViews []struct{ ByteOffset, ByteLength int }
		Meshes      []struct {
			Primitives []struct {
				Attributes map[string]int
				Indices    int
			}
		}
	}
	if err := json.Unmarshal(raw[20:20+jl], &doc); err != nil {
		t.Fatal(err)
	}
	bl := int(le.Uint32(raw[20+jl:]))
	if le.Uint32(raw[24+jl:]) != glbChunkBIN || 28+jl+bl != len(raw) {
		t.Fatal("bad BIN chunk")
	}
	bin := raw[28+jl:]
	data := func(acc int) []byte {
		v := doc.BufferViews[doc.Accessors[acc].BufferView]
		return bin[v.ByteOffset : v.ByteOffset+v.ByteLength]
	}
	pr := doc.Meshes[0].Primitives[0]
	pos := doc.Accessors[pr.Attributes["POSITION"]]
	if pos.Count != 4 || pos.Type != "VEC3" || pos.Min[0] != 0 || pos.Max[2] != 1 {
		t.Fatalf("POSITION accessor %+v", pos)
	}
	p := data(pr.Attributes["POSITION"])
	for i, v := range m.Vertices {
		if math.Float32frombits(le.Uint32(p[4*i:])) != v {
			t.Fatalf("position %d", i)
		}
	}
	ix := data(pr.Indices)
	for i, f := range m.Faces {
		if int32(le.Uint32(ix[4*i:])) != f {
			t.Fatalf("index %d", i)
		}
	}
	col := doc.Accessors[pr.Attributes["COLOR_0"]]
	if !col.Normalized || col.Type != "VEC4" || col.ComponentType != glUnsignedByte {
		t.Fatalf("COLOR_0 accessor %+v", col)
	}
	want := []byte{255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 0, 128, 255}
	if c := data(pr.Attributes["COLOR_0"]); string(c) != string(want) {
		t.Fatalf("colours % d, want % d", c, want)
	}
}
