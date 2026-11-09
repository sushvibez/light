// Tiled C[M, N] = A[M, K] . W^T for f32 A and a 16-bit (bf16 / f16)
// row-major weight W (N, K) stored two elements per u32 word: prefill-time
// nn.Linear with 16-bit weight storage (the f32 matmul's transB path with
// the weight tile widened on load).
// params: [M, N, K, mode (0 bf16, 1 f16)]
// Dispatch: (ceil(M/16), ceil(N/16), 1)

const TILE: u32 = 16u;

var<workgroup> tileA: array<f32, 256>;
var<workgroup> tileB: array<f32, 256>;

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<u32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<uniform>             params: vec4<u32>;

fn weight(i: u32, f16: bool) -> f32 {
    let word = b[i / 2u];
    let hi = (i & 1u) == 1u;
    if (f16) {
        let v = unpack2x16float(word);
        return select(v.x, v.y, hi);
    }
    return bitcast<f32>(select(word << 16u, word & 0xffff0000u, hi));
}

@compute @workgroup_size(16, 16)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id)        wid: vec3<u32>,
) {
    let M = params.x;
    let N = params.y;
    let K = params.z;
    let f16 = params.w == 1u;

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
            tileB[lx * TILE + ly] = weight(col * K + bRow, f16);
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
