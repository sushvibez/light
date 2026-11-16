// Package mesh holds triangle meshes with per-vertex colours and writes
// them as Wavefront OBJ or binary PLY.
package mesh

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// Mesh is an indexed triangle mesh: Vertices (V × 3), Faces (F × 3 vertex
// indices, counter-clockwise seen from outside) and optional per-vertex
// attributes Colors (V × ColorChannels; the first three are RGB in [0, 1]).
type Mesh struct {
	Vertices      []float32
	Faces         []int32
	Colors        []float32
	ColorChannels int
}

// NumVertices is the vertex count.
func (m *Mesh) NumVertices() int { return len(m.Vertices) / 3 }

// NumFaces is the triangle count.
func (m *Mesh) NumFaces() int { return len(m.Faces) / 3 }

// YUp returns a copy rotated from a z-up frame to the y-up one viewers use,
// (x, y, z) → (x, z, −y): TRELLIS's to_glb rotation (vertices @ [[1, 0, 0],
// [0, 0, −1], [0, 1, 0]]). Faces and colours are shared.
func (m *Mesh) YUp() *Mesh {
	out := *m
	out.Vertices = make([]float32, len(m.Vertices))
	for i := 0; i+2 < len(m.Vertices); i += 3 {
		out.Vertices[i], out.Vertices[i+1], out.Vertices[i+2] = m.Vertices[i], m.Vertices[i+2], -m.Vertices[i+1]
	}
	return &out
}

func (m *Mesh) check() error {
	if len(m.Vertices)%3 != 0 || len(m.Faces)%3 != 0 {
		return fmt.Errorf("mesh: %d vertex values, %d face indices (want multiples of 3)", len(m.Vertices), len(m.Faces))
	}
	V := m.NumVertices()
	if m.Colors != nil && (m.ColorChannels < 3 || len(m.Colors) != V*m.ColorChannels) {
		return fmt.Errorf("mesh: %d colour values for %d vertices × %d channels", len(m.Colors), V, m.ColorChannels)
	}
	for _, f := range m.Faces {
		if f < 0 || int(f) >= V {
			return fmt.Errorf("mesh: face index %d outside [0, %d)", f, V)
		}
	}
	return nil
}

// rgb returns vertex i's colour clamped to [0, 1].
func (m *Mesh) rgb(i int) [3]float32 {
	var c [3]float32
	for k := range 3 {
		c[k] = min(max(m.Colors[i*m.ColorChannels+k], 0), 1)
	}
	return c
}

// Write writes m as OBJ or PLY by the extension of path (.obj, .ply).
func Write(path string, m *Mesh) error {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".obj":
		return WriteOBJ(path, m)
	case ".ply":
		return WritePLY(path, m)
	}
	return fmt.Errorf("mesh: %s: unknown format (want .obj or .ply)", path)
}

// WriteOBJ writes m as Wavefront OBJ: "v x y z [r g b]" lines (the vertex
// colour extension MeshLab, Blender and trimesh read) and 1-based "f a b c".
func WriteOBJ(path string, m *Mesh) error {
	if err := m.check(); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	w := bufio.NewWriterSize(f, 1<<20)
	var buf []byte
	num := func(v float32) {
		buf = append(buf, ' ')
		buf = strconv.AppendFloat(buf, float64(v), 'g', -1, 32)
	}
	for i := range m.NumVertices() {
		buf = append(buf[:0], 'v')
		for k := range 3 {
			num(m.Vertices[3*i+k])
		}
		if m.Colors != nil {
			for _, c := range m.rgb(i) {
				num(c)
			}
		}
		buf = append(buf, '\n')
		w.Write(buf)
	}
	for i := range m.NumFaces() {
		buf = append(buf[:0], 'f')
		for k := range 3 {
			buf = append(buf, ' ')
			buf = strconv.AppendInt(buf, int64(m.Faces[3*i+k])+1, 10)
		}
		buf = append(buf, '\n')
		w.Write(buf)
	}
	if err := w.Flush(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// WritePLY writes m as binary little-endian PLY: float x, y, z (and uchar
// red, green, blue when coloured) per vertex, a uchar-counted int list per
// face.
func WritePLY(path string, m *Mesh) error {
	if err := m.check(); err != nil {
		return err
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	w := bufio.NewWriterSize(f, 1<<20)
	hdr := "ply\nformat binary_little_endian 1.0\ncomment written by Light\n"
	hdr += fmt.Sprintf("element vertex %d\nproperty float x\nproperty float y\nproperty float z\n", m.NumVertices())
	if m.Colors != nil {
		hdr += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
	}
	hdr += fmt.Sprintf("element face %d\nproperty list uchar int vertex_indices\nend_header\n", m.NumFaces())
	w.WriteString(hdr)
	var b [15]byte
	for i := range m.NumVertices() {
		for k := range 3 {
			binary.LittleEndian.PutUint32(b[4*k:], math.Float32bits(m.Vertices[3*i+k]))
		}
		n := 12
		if m.Colors != nil {
			for k, c := range m.rgb(i) {
				b[12+k] = uint8(math.Round(float64(c) * 255))
			}
			n = 15
		}
		w.Write(b[:n])
	}
	b[0] = 3
	for i := range m.NumFaces() {
		for k := range 3 {
			binary.LittleEndian.PutUint32(b[1+4*k:], uint32(m.Faces[3*i+k]))
		}
		w.Write(b[:13])
	}
	if err := w.Flush(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
