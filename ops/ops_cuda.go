//go:build !nocuda

package ops

// Blank-import the CUDA kernels so their init() registers with both the
// forward op dispatcher and the CNN backward hook (the latter via
// light/internal/cnnhooks — cycle-free since neither ops nor kernels/cuda
// imports the other).
import _ "light/kernels/cuda"
