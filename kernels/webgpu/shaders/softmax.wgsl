// Softmax / log-softmax along the last (contiguous) axis.
// params: [cols, rows, mode, 0]   mode: 0=softmax 1=logsoftmax
// One 256-thread workgroup per row; row = wid.x + wid.y * 65535.

@group(0) @binding(0) var<storage, read>       input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform>             params: vec4<u32>;

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

    var localMax: f32 = -3.402823e+38;
    for (var i = tid; i < cols; i = i + 256u) {
        localMax = max(localMax, input[base + i]);
    }
    shmem[tid] = localMax;
    workgroupBarrier();
    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) { shmem[tid] = max(shmem[tid], shmem[tid + s]); }
        workgroupBarrier();
    }
    let rowMax = shmem[0];
    workgroupBarrier();

    var localSum: f32 = 0.0;
    for (var i = tid; i < cols; i = i + 256u) {
        localSum = localSum + exp(input[base + i] - rowMax);
    }
    shmem[tid] = localSum;
    workgroupBarrier();
    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) { shmem[tid] = shmem[tid] + shmem[tid + s]; }
        workgroupBarrier();
    }
    let rowSum = shmem[0];
    workgroupBarrier();

    if (params.z == 0u) {
        let inv = 1.0 / rowSum;
        for (var i = tid; i < cols; i = i + 256u) {
            output[base + i] = exp(input[base + i] - rowMax) * inv;
        }
    } else {
        let lse = rowMax + log(rowSum);
        for (var i = tid; i < cols; i = i + 256u) {
            output[base + i] = input[base + i] - lse;
        }
    }
}
