// Same-shape contiguous binary elementwise op on f32.
// params: [op, len, 0, 0]   op: 0=add 1=sub 2=mul 3=div 4=pow
// Flat index = gid.x + gid.y * (65535 * 256) so dispatches larger than the
// per-dimension workgroup cap spill into y (see wgpudev.Grid1D).

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             params: vec4<u32>;

const Y_STRIDE: u32 = 65535u * 256u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * Y_STRIDE;
    if (idx >= params.y) { return; }
    let x = a[idx];
    let y = b[idx];
    switch params.x {
        case 0u: { out[idx] = x + y; }
        case 1u: { out[idx] = x - y; }
        case 2u: { out[idx] = x * y; }
        case 3u: { out[idx] = x / y; }
        case 4u: { out[idx] = pow(x, y); }
        default: {}
    }
}
