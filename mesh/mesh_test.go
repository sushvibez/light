package mesh

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func tetra() *Mesh {
	return &Mesh{
		Vertices:      []float32{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1},
		Faces:         []int32{0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3},
		Colors:        []float32{1, 0, 0, 9, 9, 9, 0, 1, 0, 9, 9, 9, 0, 0, 1, 9, 9, 9, 1.5, -1, 0.5, 9, 9, 9},
		ColorChannels: 6,
	}
}

func TestWriteOBJ(t *testing.T) {
	path := filepath.Join(t.TempDir(), "t.obj")
	if err := Write(path, tetra()); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	want := "v 0 0 0 1 0 0\nv 1 0 0 0 1 0\nv 0 1 0 0 0 1\nv 0 0 1 1 0 0.5\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n"
	if string(raw) != want {
		t.Errorf("OBJ:\n%s\nwant\n%s", raw, want)
	}
}

func TestWritePLY(t *testing.T) {
	path := filepath.Join(t.TempDir(), "t.ply")
	m := tetra()
	if err := Write(path, m); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	end := []byte("end_header\n")
	i := bytes.Index(raw, end)
	if i < 0 || !strings.Contains(string(raw[:i]), "element vertex 4\n") || !strings.Contains(string(raw[:i]), "element face 4\n") {
		t.Fatalf("header %q", raw[:max(i, 0)])
	}
	body := raw[i+len(end):]
	if len(body) != 4*15+4*13 {
		t.Fatalf("%d body bytes, want %d", len(body), 4*15+4*13)
	}
	if x := math.Float32frombits(binary.LittleEndian.Uint32(body[15:])); x != 1 {
		t.Errorf("vertex 1 x = %v", x)
	}
	if c := body[3*15+12 : 3*15+15]; !bytes.Equal(c, []byte{255, 0, 128}) {
		t.Errorf("vertex 3 colour %v, want clamped [255 0 128]", c)
	}
	f := body[60+13:]
	if f[0] != 3 || binary.LittleEndian.Uint32(f[1:]) != 0 || binary.LittleEndian.Uint32(f[5:]) != 1 || binary.LittleEndian.Uint32(f[9:]) != 3 {
		t.Errorf("face 1 %v", f[:13])
	}
}

func TestYUpAndChecks(t *testing.T) {
	m := tetra().YUp()
	if v := m.Vertices[9:12]; v[0] != 0 || v[1] != 1 || v[2] != 0 {
		t.Errorf("(0, 0, 1) → %v, want (0, 1, 0)", v)
	}
	if v := m.Vertices[6:9]; v[0] != 0 || v[1] != 0 || v[2] != -1 {
		t.Errorf("(0, 1, 0) → %v, want (0, 0, -1)", v)
	}
	bad := tetra()
	bad.Faces[0] = 4
	if err := Write(filepath.Join(t.TempDir(), "b.obj"), bad); err == nil {
		t.Error("face index 4 of 4 vertices accepted")
	}
	if err := Write(filepath.Join(t.TempDir(), "b.stl"), tetra()); err == nil {
		t.Error(".stl accepted")
	}
}
