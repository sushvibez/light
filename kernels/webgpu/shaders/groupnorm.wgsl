// GroupNorm over (N, C, H, W): each (n, g) is a contiguous block of
// M = (C/G)·H·W values (row = n·G + g), normalised by its mean and variance
// (two passes) then scaled per channel: c = g·(C/G) + i / (H·W).
// params: [M, HW, eps_bits, G]
// One workgroup per (n, g); row = wid.x + wid.y * 65535.

@group(0) @binding(0) var<storage, read>       input: array<f32>;
@group(0) @binding(1) var<storage, read>       gamma: array<f32>;
@group(0) @binding(2) var<storage, read>       beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform>             params: vec4<u32>;

var<workgroup> shmem: array<f32, 256>;

fn reduce(tid: u32, v: f32) -> f32 {
    shmem[tid] = v;
    workgroupBarrier();
    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) { shmem[tid] = shmem[tid] + shmem[tid + s]; }
        workgroupBarrier();
    }
    let r = shmem[0];
    workgroupBarrier();
    return r;
}

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id)        wid: vec3<u32>,
) {
    let M = params.x;
    let HW = params.y;
    let eps = bitcast<f32>(params.z);
    let G = params.w;
    let row = wid.x + wid.y * 65535u;
    if (row * M >= arrayLength(&output)) { return; }
    let tid = lid.x;
    let base = row * M;
    let c0 = (row % G) * (M / HW);

    var s: f32 = 0.0;
    for (var i = tid; i < M; i = i + 256u) { s = s + input[base + i]; }
    let mean = reduce(tid, s) / f32(M);
    var v: f32 = 0.0;
    for (var i = tid; i < M; i = i + 256u) {
        let d = input[base + i] - mean;
        v = v + d * d;
    }
    let invStd = 1.0 / sqrt(reduce(tid, v) / f32(M) + eps);
    for (var i = tid; i < M; i = i + 256u) {
        let c = c0 + i / HW;
        output[base + i] = (input[base + i] - mean) * invStd * gamma[c] + beta[c];
    }
}
