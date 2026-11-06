// Reduce each row of a [rows, cols] contiguous f32 matrix to one value.
// params: [op, rows, cols, 0]   op: 0=sum 1=mean 2=max 3=min 4=prod
// One workgroup per row; row = wid.x + wid.y * 65535.

@group(0) @binding(0) var<storage, read>       input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform>             params: vec4<u32>;

var<workgroup> shmem: array<f32, 256>;

fn identity(op: u32) -> f32 {
    switch op {
        case 2u: { return -3.402823e+38; }
        case 3u: { return 3.402823e+38; }
        case 4u: { return 1.0; }
        default: { return 0.0; }
    }
}

fn combine(op: u32, x: f32, y: f32) -> f32 {
    switch op {
        case 2u: { return max(x, y); }
        case 3u: { return min(x, y); }
        case 4u: { return x * y; }
        default: { return x + y; }
    }
}

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id)        wid: vec3<u32>,
) {
    let op = params.x;
    let cols = params.z;
    let row = wid.x + wid.y * 65535u;
    if (row >= params.y) { return; }
    let tid = lid.x;
    let base = row * cols;

    var acc = identity(op);
    for (var i = tid; i < cols; i = i + 256u) {
        acc = combine(op, acc, input[base + i]);
    }
    shmem[tid] = acc;
    workgroupBarrier();
    for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) { shmem[tid] = combine(op, shmem[tid], shmem[tid + s]); }
        workgroupBarrier();
    }
    if (tid == 0u) {
        var r = shmem[0];
        if (op == 1u) { r = r / f32(cols); }
        output[row] = r;
    }
}
