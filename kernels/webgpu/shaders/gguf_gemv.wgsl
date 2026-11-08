// y[m, n] = Σ_k x[m, k] · W[n, k] for M ≤ 8 rows, decoding W's blocks as
// they are read (decode-time matmul: the weight is read once, quantized).
// One workgroup per output column n; its 64 threads take 32-element chunks
// of the row (a Q*_0 block, or a sub-block of a K-quant super-block), so
// each chunk's scale fields are decoded once, not once per element.
// params: [M, N, K, 0]

@group(0) @binding(1) var<storage, read>       x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform>             params: vec4<u32>;

const WG: u32 = 64u;
const MAXM: u32 = 8u;
var<workgroup> partial: array<f32, 512>; // WG × MAXM

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let n = wid.x + wid.y * 65535u;
    let M = params.x;
    let N = params.y;
    let K = params.z;
    if (n >= N) { return; }
    var acc: array<f32, 8>;
    for (var m = 0u; m < MAXM; m++) { acc[m] = 0.0; }
    let subs = BE / 32u;
    let rowBase = n * (K / BE) * BB;
    for (var c = lid.x; c < K / 32u; c += WG) {
        let base = rowBase + (c / subs) * BB;
        let sub = c % subs;
        let k0 = c * 32u;
        // Per format: decode the chunk's scale fields once, then each value
        // straight into the accumulators (no per-chunk scratch array).
        switch QTYPE {
            case 8u: { // Q8_0
                let d = f16_at(base);
                for (var i = 0u; i < 32u; i++) {
                    let v = d * i8_at(base + 2u + i);
                    for (var m = 0u; m < M; m++) { acc[m] += x[m * K + k0 + i] * v; }
                }
            }
            case 2u, 6u: { // Q4_0, Q5_0 (fifth bit i of the u32 at base + 2)
                let d = f16_at(base);
                let qo = select(2u, 6u, QTYPE == 6u);
                for (var i = 0u; i < 32u; i++) {
                    let q = byte_at(base + qo + (i & 15u));
                    var nib = select(q & 15u, q >> 4u, i >= 16u);
                    var off = 8;
                    if (QTYPE == 6u) {
                        nib = nib | (((byte_at(base + 2u + (i >> 3u)) >> (i & 7u)) & 1u) << 4u);
                        off = 16;
                    }
                    let v = d * f32(i32(nib) - off);
                    for (var m = 0u; m < M; m++) { acc[m] += x[m * K + k0 + i] * v; }
                }
            }
            case 12u: { // Q4_K: sub-block s = 2p + hi
                let p = sub / 2u;
                let hi = (sub & 1u) == 1u;
                let smin = scale_min_k4(sub, base + 4u);
                let dl = f16_at(base) * smin.x;
                let ml = f16_at(base + 2u) * smin.y;
                for (var l = 0u; l < 32u; l++) {
                    let q = byte_at(base + 16u + p * 32u + l);
                    let v = dl * f32(select(q & 15u, q >> 4u, hi)) - ml;
                    for (var m = 0u; m < M; m++) { acc[m] += x[m * K + k0 + l] * v; }
                }
            }
            case 14u: { // Q6_K: sub-block s = 4·half + quarter
                let half = sub / 4u;
                let quarter = sub % 4u;
                let d = f16_at(base + 208u);
                let sb = base + 192u + half * 8u + 2u * quarter;
                let s0 = d * i8_at(sb);
                let s1 = d * i8_at(sb + 1u);
                let ql = base + half * 64u + select(0u, 32u, (quarter & 1u) == 1u);
                let qh = base + 128u + half * 32u;
                for (var l = 0u; l < 32u; l++) {
                    let lo = byte_at(ql + l);
                    let nib = select(lo & 15u, lo >> 4u, quarter >= 2u);
                    let raw = nib | (((byte_at(qh + l) >> (2u * quarter)) & 3u) << 4u);
                    let v = select(s0, s1, l >= 16u) * f32(i32(raw) - 32);
                    for (var m = 0u; m < M; m++) { acc[m] += x[m * K + k0 + l] * v; }
                }
            }
            default: {}
        }
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
