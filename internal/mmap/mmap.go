// Package mmap maps files read-only: the pages stay file-backed, so the OS
// can drop and re-read them instead of counting them against memory (a
// copy-on-write view would be charged in full as private memory on
// Windows). Writing to Data faults. Loaders use it to build tensors
// over checkpoint bytes without first reading the whole file into the heap.
package mmap

import (
	"errors"
	"os"
	"sync"
)

// ErrUnsupported is returned where mapping is not available (the caller
// reads the file instead).
var ErrUnsupported = errors.New("mmap: not supported on this platform")

// File is a mapped file. Data is valid until Close.
type File struct {
	Data  []byte
	once  sync.Once
	unmap func() error
}

// Open maps path read-only. An empty file maps to an empty Data.
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if st.Size() == 0 {
		return &File{unmap: func() error { return nil }}, nil
	}
	data, unmap, err := mapFile(f, st.Size())
	if err != nil {
		return nil, err
	}
	return &File{Data: data, unmap: unmap}, nil
}

// Close unmaps the file; Data must not be used afterwards. Safe to call
// more than once.
func (m *File) Close() error {
	var err error
	m.once.Do(func() {
		err = m.unmap()
		m.Data = nil
	})
	return err
}
