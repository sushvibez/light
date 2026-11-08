// out[offset + r·K + k] = row (ids? ids[r] : r), element k, dequantized.
// params: [rows, K, mode (0: rows in order, 1: gather I64 ids, 2: I32 ids), offset]

@group(0) @binding(1) var<storage, read>       ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             params: vec4<u32>;

const Y_STRIDE: u32 = 65535u * 256u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * Y_STRIDE;
    let K = params.y;
    if (idx >= params.x * K) { return; }
    let r = idx / K;
    let k = idx % K;
    var n = r;
    if (params.z == 1u) { n = ids[r * 2u]; }
    if (params.z == 2u) { n = ids[r]; }
    out[params.w + idx] = row_elem(n, K, k);
}
