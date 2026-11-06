// RMSNorm over the last axis: y = x * rsqrt(mean(x^2) + eps) * weight
// params: [cols, rows, eps_bits, 0]

@group(0) @binding(0) var<storage, read>       input: array<f32>;
@group(0) @binding(1) var<storage, read>       weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform>             params: vec4<u32>;

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

    var localSumSq: f32 = 0.0;
    for (var i = tid; i < cols; i = i + 256u) {
        let v = input[base + i];
        localSumSq = localSumSq + v * v;
    }
    shmem[tid] = localSumSq;
    workgroupBarrier();
    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) { shmem[tid] = shmem[tid] + shmem[tid + s]; }
        workgroupBarrier();
    }
    let invRms = 1.0 / sqrt(shmem[0] / f32(cols) + eps);
    workgroupBarrier();

    for (var i = tid; i < cols; i = i + 256u) {
        output[base + i] = input[base + i] * invRms * weight[i];
    }
}
