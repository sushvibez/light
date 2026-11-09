//go:build !windows && !unix

package mmap

import "os"

func mapFile(*os.File, int64) ([]byte, func() error, error) {
	return nil, nil, ErrUnsupported
}
