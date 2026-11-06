//go:build !nocuda

package cudart_test

import "os"

// osReadFile is defined here to avoid an extra import in the main test file.
func osReadFile(path string) ([]byte, error) { return os.ReadFile(path) }
