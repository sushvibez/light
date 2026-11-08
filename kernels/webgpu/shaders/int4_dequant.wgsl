// Canonical 4-bit weights (quant.Int4) → f32: codes (N, K/8) u32 with
// nibble j of word w holding k = 8w + j; scales / zeros (N, K/G);
// W[n, k] = s · (code − z), written at out[offset + n·K + k].
// params: [N, K, group, offset]

@group(0) @binding(0) var<storage, read>       codes: array<u32>;
@group(0) @binding(1) var<storage, read>       scales: array<f32>;
@group(0) @binding(2) var<storage, read>       zeros: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform>             params: vec4<u32>;

const Y_STRIDE: u32 = 65535u * 256u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * Y_STRIDE;
    let K = params.y;
    if (idx >= params.x * K) { return; }
    let n = idx / K;
    let k = idx % K;
    let g = n * (K / params.z) + k / params.z;
    let code = (codes[n * (K / 8u) + k / 8u] >> (4u * (k % 8u))) & 15u;
    out[params.w + idx] = scales[g] * (f32(code) - zeros[g]);
}
