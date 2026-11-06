// Gather a strided view into a packed row-major buffer. Operates on 32-bit
// words, so it serves every 4-byte dtype directly; 8-byte dtypes are
// handled by the launcher by appending a trailing axis of size 2.
// shp: [dims[rank], srcStrides[rank]] (i32, word units)
// params: [len, rank, srcOffset(words), 0]

@group(0) @binding(0) var<storage, read>       src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;
@group(0) @binding(2) var<storage, read>       shp: array<i32>;
@group(0) @binding(3) var<uniform>             params: vec4<u32>;

const Y_STRIDE: u32 = 65535u * 256u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * Y_STRIDE;
    if (idx >= params.x) { return; }
    let rank = params.y;
    var rem = i32(idx);
    var off = i32(params.z);
    for (var k: u32 = 0u; k < rank; k = k + 1u) {
        let ax = rank - 1u - k;
        let d = shp[ax];
        let c = rem % d;
        rem = rem / d;
        off = off + c * shp[rank + ax];
    }
    dst[idx] = src[u32(off)];
}
