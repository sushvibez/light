package gpu

// Limits are the buffer-size limits a device was created with. A device
// is requested with its adapter's own maxBufferSize and
// maxStorageBufferBindingSize, so on a desktop GPU these run to gigabytes;
// the constants below are WebGPU's guaranteed minimums, what a device
// falls back to when the adapter cannot be asked.
type Limits struct {
	MaxBufferSize         int64
	MaxStorageBindingSize int64
}

// MaxBufferSize is the largest single buffer WebGPU guarantees, and what
// wgpu-native enforces when a device is requested without limits. A request past it does not fail
// usefully: the driver logs an uncaptured validation error, hands back an
// invalid handle, and the program goes on using it until a draw command
// aborts the process inside Rust, hundreds of frames and hundreds of
// identical error lines later, with no Go stack to say who asked.
const MaxBufferSize = 256 << 20

// MaxStorageBindingSize is the largest storage buffer a bind group may
// bind whole — WebGPU's default limit, half of MaxBufferSize. A storage
// buffer that grows past it is a valid buffer that cannot be bound.
const MaxStorageBindingSize = 128 << 20
