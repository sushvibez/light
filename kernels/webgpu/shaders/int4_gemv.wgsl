// y[m, n] = Σ_k x[m, k] · s[n, g(k)] · (code[n, k] − z[n, g(k)]) for M ≤ 8
// activation rows, straight from the 4-bit codes (decode: the weight is
// read once, packed). One workgroup per output column n; its 64 threads
// stride over the packed words of row n and reduce in shared memory.
// Requires K % 8 == 0 and group % 8 == 0 (a word never straddles groups).
// params: [M, N, K, group]

@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read>       codes: array<u32>;
@group(0) @binding(2) var<storage, read>       scales: array<f32>;
@group(0) @binding(3) var<storage, read>       zeros: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform>             params: vec4<u32>;

const WG: u32 = 64u;
const MAXM: u32 = 8u;
var<workgroup> partial: array<f32, 512>; // WG × MAXM

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n = wid.x + wid.y * 65535u;
    let M = params.x;
    let N = params.y;
    let K = params.z;
    let G = params.w;
    if (n >= N) { return; }
    let words = K / 8u;
    let ng = K / G;
    var acc: array<f32, 8>;
    for (var m = 0u; m < MAXM; m++) { acc[m] = 0.0; }
    for (var w = lid.x; w < words; w += WG) {
        let word = codes[n * words + w];
        let k0 = w * 8u;
        let g = n * ng + k0 / G;
        let s = scales[g];
        let z = zeros[g];
        for (var j = 0u; j < 8u; j++) {
            let wv = s * (f32((word >> (4u * j)) & 15u) - z);
            for (var m = 0u; m < M; m++) {
                acc[m] += x[m * K + k0 + j] * wv;
            }
        }
    }
    for (var m = 0u; m < MAXM; m++) { partial[m * WG + lid.x] = acc[m]; }
    workgroupBarrier();
    for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
        if (lid.x < stride) {
            for (var m = 0u; m < M; m++) {
                partial[m * WG + lid.x] += partial[m * WG + lid.x + stride];
            }
        }
        workgroupBarrier();
    }
    if (lid.x == 0u) {
        for (var m = 0u; m < M; m++) { out[m * N + n] = partial[m * WG]; }
    }
}
