// LayerNorm over the last axis: y = (x - mean) / sqrt(var + eps) * gamma + beta
// params: [cols, rows, eps_bits, 0]
// One workgroup per row; row = wid.x + wid.y * 65535.

@group(0) @binding(0) var<storage, read>       input: array<f32>;
@group(0) @binding(1) var<storage, read>       gamma: array<f32>;
@group(0) @binding(2) var<storage, read>       beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform>             params: vec4<u32>;

var<workgroup> shmem: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id)        wid: vec3<u32>,
) {
    let cols = params.x;
    let row = wid.x + wid.y * 65535u;
    if (row >= params.y) { return; }
    let tid = lid.x;
    let base = row * cols;
    let eps = bitcast<f32>(params.z);

    var localSum: f32 = 0.0;
    for (var i = tid; i < cols; i = i + 256u) {
        localSum = localSum + input[base + i];
    }
    shmem[tid] = localSum;
    workgroupBarrier();
    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) { shmem[tid] = shmem[tid] + shmem[tid + s]; }
        workgroupBarrier();
    }
    let mean = shmem[0] / f32(cols);
    workgroupBarrier();

    var localVar: f32 = 0.0;
    for (var i = tid; i < cols; i = i + 256u) {
        let d = input[base + i] - mean;
        localVar = localVar + d * d;
    }
    shmem[tid] = localVar;
    workgroupBarrier();
    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) { shmem[tid] = shmem[tid] + shmem[tid + s]; }
        workgroupBarrier();
    }
    let variance = shmem[0] / f32(cols);
    let invStd = 1.0 / sqrt(variance + eps);
    workgroupBarrier();

    for (var i = tid; i < cols; i = i + 256u) {
        output[base + i] = (input[base + i] - mean) * invStd * gamma[i] + beta[i];
    }
}
