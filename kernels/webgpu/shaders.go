//go:build !nowebgpu

package webgpu

import (
	"embed"
	"fmt"
)

//go:embed shaders/*.wgsl
var shaderFS embed.FS

// shader returns the embedded WGSL source for name (without extension).
func shader(name string) string {
	b, err := shaderFS.ReadFile("shaders/" + name + ".wgsl")
	if err != nil {
		panic(fmt.Sprintf("webgpu: missing embedded shader %q: %v", name, err))
	}
	return string(b)
}
