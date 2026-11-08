package gpu

// BytesPerTexel is a format's size in bytes: an upload's row stride and
// ReadTexture's. Formats it does not list are taken as four bytes.
func BytesPerTexel(f TextureFormat) int {
	switch f {
	case FormatR8Unorm:
		return 1
	case FormatRGBA16Float:
		return 8
	}
	return 4
}

// The formats by their WebGPU names. TextureFormat is a number natively
// (WGPUTextureFormat) and the name itself on the web, so a literal works
// on one build only; code that has a format as text (a config file, a
// flag) converts it through FormatByName and FormatName instead.
var formatNames = []struct {
	f    TextureFormat
	name string
}{
	{FormatRGBA8Unorm, "rgba8unorm"},
	{FormatBGRA8Unorm, "bgra8unorm"},
	{FormatR8Unorm, "r8unorm"},
	{FormatRGBA16Float, "rgba16float"},
	{FormatDepth32Float, "depth32float"},
}

// FormatName is a format's WebGPU name ("rgba8unorm"), or "" for a format
// this package has no constant for.
func FormatName(f TextureFormat) string {
	for _, e := range formatNames {
		if e.f == f {
			return e.name
		}
	}
	return ""
}

// FormatByName is the format with that WebGPU name, the same value on
// every build.
func FormatByName(name string) (TextureFormat, bool) {
	for _, e := range formatNames {
		if e.name == name {
			return e.f, true
		}
	}
	var zero TextureFormat
	return zero, false
}

// VertexStepMode says what a vertex buffer's records are indexed by.
type VertexStepMode uint32

// WGPUVertexStepMode (webgpu.h): Undefined is 0, and a layout that
// leaves it there with attributes set defaults to Vertex — which is why
// every layout written before this existed still steps per vertex.
const (
	StepUndefined VertexStepMode = 0
	StepVertex    VertexStepMode = 1
	StepInstance  VertexStepMode = 2
)
