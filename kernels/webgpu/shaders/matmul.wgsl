// Tiled matmul: C[M,N] = A[M,K] @ B, with B either row-major [K,N]
// (transB=0) or the row-major [N,K] weight read transposed (transB=1), so
// nn.Linear's W^T view never has to be materialised.
// params: [M, N, K, transB]
// Dispatch: (ceil(M/16), ceil(N/16), 1)

const TILE: u32 = 16u;

var<workgroup> tileA: array<f32, 256>;
var<workgroup> tileB: array<f32, 256>;

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<uniform>             params: vec4<u32>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id)        wid: vec3<u32>,
) {
    let M = params.x;
    let N = params.y;
    let K = params.z;
    let transB = params.w;

    let row = wid.x * TILE + lid.x;
    let col = wid.y * TILE + lid.y;
    let lx = lid.x;
    let ly = lid.y;

    var acc: f32 = 0.0;
    let numTiles = (K + TILE - 1u) / TILE;

    for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
        let aCol = t * TILE + ly;
        if (row < M && aCol < K) {
            tileA[lx * TILE + ly] = a[row * K + aCol];
        } else {
            tileA[lx * TILE + ly] = 0.0;
        }

        let bRow = t * TILE + lx;
        if (bRow < K && col < N) {
            if (transB == 0u) {
                tileB[lx * TILE + ly] = b[bRow * N + col];
            } else {
                tileB[lx * TILE + ly] = b[col * K + bRow];
            }
        } else {
            tileB[lx * TILE + ly] = 0.0;
        }

        workgroupBarrier();
        for (var k: u32 = 0u; k < TILE; k = k + 1u) {
            acc = acc + tileA[lx * TILE + k] * tileB[k * TILE + ly];
        }
        workgroupBarrier();
    }

    if (row < M && col < N) {
        result[row * N + col] = acc;
    }
}
