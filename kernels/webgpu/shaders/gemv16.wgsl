// y[m, n] = sum_k x[m, k] * w[n, k] for M <= 8 rows of f32 x and a
// 16-bit (bf16 or f16) row-major weight w (N, K), two elements per u32
// word: decode-time nn.Linear with 16-bit weight storage. One workgroup
// per output column; 64 threads stride over the row's words (K even).
// params: [M, N, K, mode (0 bf16, 1 f16)]

@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read>       w: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             params: vec4<u32>;

const WG: u32 = 64u;
const MAXM: u32 = 8u;
var<workgroup> partial: array<f32, 512>; // WG x MAXM

fn widen(word: u32, f16: bool) -> vec2<f32> {
    if (f16) { return unpack2x16float(word); }
    return vec2<f32>(bitcast<f32>(word << 16u), bitcast<f32>(word & 0xffff0000u));
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n = wid.x + wid.y * 65535u;
    let M = params.x;
    let N = params.y;
    let K = params.z;
    let f16 = params.w == 1u;
    if (n >= N) { return; }
    var acc: array<f32, 8>;
    for (var m = 0u; m < MAXM; m++) { acc[m] = 0.0; }
    let words = K / 2u;
    let row = n * words;
    for (var i = lid.x; i < words; i += WG) {
        let v = widen(w[row + i], f16);
        let k = 2u * i;
        for (var m = 0u; m < M; m++) { acc[m] += x[m * K + k] * v.x + x[m * K + k + 1u] * v.y; }
    }
    for (var m = 0u; m < MAXM; m++) { partial[m * WG + lid.x] = acc[m]; }
    workgroupBarrier();
    for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
        if (lid.x < stride) {
            for (var m = 0u; m < M; m++) { partial[m * WG + lid.x] += partial[m * WG + lid.x + stride]; }
        }
        workgroupBarrier();
    }
    if (lid.x == 0u) {
        for (var m = 0u; m < M; m++) { out[m * N + n] = partial[m * WG]; }
    }
}
