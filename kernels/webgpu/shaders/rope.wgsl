// Rotary position embedding on contiguous f32 (..., seq, headDim).
// One invocation per (row, i < headDim/2); row = flattened (..., seq).
// theta = pos[seq index] * invFreq[i] (invFreq computed on the host in f64).
// mode 0: rotate-half pairs (i, i + half); mode 1: interleaved (2i, 2i + 1).
// params: [headDim, seqLen, rows, mode]

@group(0) @binding(0) var<storage, read>       x: array<f32>;
@group(0) @binding(1) var<storage, read>       pos: array<u32>;   // I64 positions: low words
@group(0) @binding(2) var<storage, read>       invf: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;
@group(0) @binding(4) var<uniform>             params: vec4<u32>;

const Y_STRIDE: u32 = 65535u * 256u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * Y_STRIDE;
    let hd = params.x;
    let half = hd / 2u;
    if (idx >= params.z * half) { return; }
    let row = idx / half;
    let i = idx % half;
    let p = f32(pos[(row % params.y) * 2u]);
    let theta = p * invf[i];
    let c = cos(theta);
    let s = sin(theta);
    var a = row * hd + i;
    var b = a + half;
    if (params.w == 1u) {
        a = row * hd + 2u * i;
        b = a + 1u;
    }
    let x0 = x[a];
    let x1 = x[b];
    out[a] = x0 * c - x1 * s;
    out[b] = x0 * s + x1 * c;
}
