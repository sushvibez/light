// y[m, n] = sum_k x[m, k] * w[n, k] for M <= 8 rows of x and a row-major
// weight w (N, K): nn.Linear's x . W^T at decode. One workgroup per output
// column n; 64 threads stride over k (adjacent threads read adjacent
// weights) and reduce in shared memory. The tiled matmul leaves most of
// its 16 x 16 threads idle when M is 1.
// params: [M, N, K, 0]

@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read>       w: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             params: vec4<u32>;

const WG: u32 = 64u;
const MAXM: u32 = 8u;
var<workgroup> partial: array<f32, 512>; // WG x MAXM

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n = wid.x + wid.y * 65535u;
    let M = params.x;
    let N = params.y;
    let K = params.z;
    if (n >= N) { return; }
    var acc: array<f32, 8>;
    for (var m = 0u; m < MAXM; m++) { acc[m] = 0.0; }
    let row = n * K;
    for (var k = lid.x; k < K; k += WG) {
        let wv = w[row + k];
        for (var m = 0u; m < M; m++) { acc[m] += x[m * K + k] * wv; }
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
