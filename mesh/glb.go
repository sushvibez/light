package mesh

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// glTF 2.0 constants used by WriteGLB.
const (
	glbMagic        = 0x46546C67 // "glTF"
	glbChunkJSON    = 0x4E4F534A
	glbChunkBIN     = 0x004E4942
	glFloat         = 5126
	glUnsignedByte  = 5121
	glUnsignedInt   = 5125
	glArrayBuffer   = 34962
	glElementBuffer = 34963
)

// WriteGLB writes m as a binary glTF 2.0 file: one triangle primitive with
// float POSITION, uint32 indices and, when coloured, normalised uint8 RGBA
// COLOR_0 (the colours as stored, as trimesh's export writes them) and no
// material (glTF's default white one: viewers show the vertex colours, and
// trimesh loads them as vertex colours).
func WriteGLB(path string, m *Mesh) error {
	b, err := EncodeGLB(m)
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

// EncodeGLB returns WriteGLB's bytes.
func EncodeGLB(m *Mesh) ([]byte, error) {
	if err := m.check(); err != nil {
		return nil, err
	}
	V, F := m.NumVertices(), m.NumFaces()
	if V == 0 || F == 0 {
		return nil, fmt.Errorf("mesh: GLB of an empty mesh (%d vertices, %d faces)", V, F)
	}
	var bin bytes.Buffer
	le := binary.LittleEndian
	// Positions, then indices, then colours: every view starts 4-aligned.
	lo := [3]float64{math.Inf(1), math.Inf(1), math.Inf(1)}
	hi := [3]float64{math.Inf(-1), math.Inf(-1), math.Inf(-1)}
	for i, v := range m.Vertices {
		lo[i%3], hi[i%3] = math.Min(lo[i%3], float64(v)), math.Max(hi[i%3], float64(v))
		binary.Write(&bin, le, v)
	}
	posLen := bin.Len()
	for _, f := range m.Faces {
		binary.Write(&bin, le, uint32(f))
	}
	idxLen := bin.Len() - posLen
	type view struct {
		Buffer     int `json:"buffer"`
		ByteOffset int `json:"byteOffset"`
		ByteLength int `json:"byteLength"`
		Target     int `json:"target"`
	}
	type accessor struct {
		BufferView    int       `json:"bufferView"`
		ComponentType int       `json:"componentType"`
		Normalized    bool      `json:"normalized,omitempty"`
		Count         int       `json:"count"`
		Type          string    `json:"type"`
		Min           []float64 `json:"min,omitempty"`
		Max           []float64 `json:"max,omitempty"`
	}
	views := []view{{0, 0, posLen, glArrayBuffer}, {0, posLen, idxLen, glElementBuffer}}
	accs := []accessor{
		{BufferView: 0, ComponentType: glFloat, Count: V, Type: "VEC3", Min: lo[:], Max: hi[:]},
		{BufferView: 1, ComponentType: glUnsignedInt, Count: 3 * F, Type: "SCALAR"},
	}
	attrs := map[string]int{"POSITION": 0}
	if m.Colors != nil {
		off := bin.Len()
		for i := range V {
			c := m.rgb(i)
			bin.Write([]byte{uint8(math.Round(float64(c[0]) * 255)), uint8(math.Round(float64(c[1]) * 255)), uint8(math.Round(float64(c[2]) * 255)), 255})
		}
		views = append(views, view{0, off, bin.Len() - off, glArrayBuffer})
		accs = append(accs, accessor{BufferView: 2, ComponentType: glUnsignedByte, Normalized: true, Count: V, Type: "VEC4"})
		attrs["COLOR_0"] = 2
	}
	doc := map[string]any{
		"asset":       map[string]any{"version": "2.0", "generator": "light"},
		"scene":       0,
		"scenes":      []any{map[string]any{"nodes": []int{0}}},
		"nodes":       []any{map[string]any{"mesh": 0}},
		"meshes":      []any{map[string]any{"primitives": []any{map[string]any{"attributes": attrs, "indices": 1, "mode": 4}}}},
		"buffers":     []any{map[string]any{"byteLength": bin.Len()}},
		"bufferViews": views,
		"accessors":   accs,
	}
	js, err := json.Marshal(doc)
	if err != nil {
		return nil, err
	}
	for len(js)%4 != 0 {
		js = append(js, ' ')
	}
	for bin.Len()%4 != 0 {
		bin.WriteByte(0)
	}
	var out bytes.Buffer
	total := 12 + 8 + len(js) + 8 + bin.Len()
	for _, w := range []uint32{glbMagic, 2, uint32(total), uint32(len(js)), glbChunkJSON} {
		binary.Write(&out, le, w)
	}
	out.Write(js)
	binary.Write(&out, le, uint32(bin.Len()))
	binary.Write(&out, le, uint32(glbChunkBIN))
	out.Write(bin.Bytes())
	return out.Bytes(), nil
}
