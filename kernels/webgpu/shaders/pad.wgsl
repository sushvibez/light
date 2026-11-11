// Zero padding along one axis of a contiguous tensor viewed as
// (outer, len, inner): out[o, t, i] = in[o, t - left, i] inside the
// source range, 0 outside.
// params: [outLen, inner, left, inLen]; one invocation per output element.

@group(0) @binding(0) var<storage, read>       input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform>             params: vec4<u32>;

const Y_STRIDE: u32 = 65535u * 256u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * Y_STRIDE;
    if (idx >= arrayLength(&output)) { return; }
    let outLen = params.x;
    let inner = params.y;
    let left = params.z;
    let inLen = params.w;
    let i = idx % inner;
    let t = (idx / inner) % outLen;
    let o = idx / (inner * outLen);
    if (t < left || t >= left + inLen) {
        output[idx] = 0.0;
        return;
    }
    output[idx] = input[(o * inLen + (t - left)) * inner + i];
}
