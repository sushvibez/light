package mmap

import (
	"os"
	"path/filepath"
	"testing"
)

// The mapping reads the file and closes cleanly (twice).
func TestOpen(t *testing.T) {
	p := filepath.Join(t.TempDir(), "f")
	want := []byte("hello, mapped world")
	if err := os.WriteFile(p, want, 0o644); err != nil {
		t.Fatal(err)
	}
	m, err := Open(p)
	if err != nil {
		t.Fatal(err)
	}
	if string(m.Data) != string(want) {
		t.Fatalf("mapped %q, want %q", m.Data, want)
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
	if err := m.Close(); err != nil {
		t.Fatal(err)
	}
}
