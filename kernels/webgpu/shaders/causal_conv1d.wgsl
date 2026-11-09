// Causal depthwise conv1d (Gated DeltaNet's short conv): out[b, c, t] =
// sum_j w[c, j] * in(t - K + 1 + j), in() reading the K-1 carried-over
// inputs (state) before t = 0, optional SiLU; the new state is the last
// K-1 inputs. One invocation per (b, c) row.
// params: [rows = B*C, C, T, K | silu << 16]

@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read>       w: array<f32>;
@group(0) @binding(2) var<storage, read>       st: array<f32>;
@group(0) @binding(3) var<storage, read_write> nst: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform>             params: vec4<u32>;

const Y_STRIDE: u32 = 65535u * 256u;

fn input(row: u32, i: i32, T: u32, P: u32) -> f32 {
    if (i < 0) { return st[row * P + u32(i32(P) + i)]; }
    return x[row * T + u32(i)];
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x + gid.y * Y_STRIDE;
    if (row >= params.x) { return; }
    let c = row % params.y;
    let T = params.z;
    let K = params.w & 0xFFFFu;
    let silu = (params.w >> 16u) != 0u;
    let P = K - 1u;
    for (var t = 0u; t < T; t++) {
        var acc = 0.0;
        for (var j = 0u; j < K; j++) {
            acc += w[c * K + j] * input(row, i32(t) - i32(P) + i32(j), T, P);
        }
        if (silu) { acc = acc / (1.0 + exp(-acc)); }
        out[row * T + t] = acc;
    }
    for (var i = 0u; i < P; i++) {
        nst[row * P + i] = input(row, i32(T) - i32(P) + i32(i), T, P);
    }
}
