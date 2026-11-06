// Row gather along axis 0, on 32-bit words (dtype-agnostic for 4/8-byte
// element types; the launcher scales rowWords accordingly).
// params: [numIndices, rowWords, idxStride, 0]
//   idxStride = 2 for I64 indices (low word read), 1 for I32.

@group(0) @binding(0) var<storage, read>       src: array<u32>;
@group(0) @binding(1) var<storage, read>       indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> out: array<u32>;
@group(0) @binding(3) var<uniform>             params: vec4<u32>;

const Y_STRIDE: u32 = 65535u * 256u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * Y_STRIDE;
    let rowWords = params.y;
    let total = params.x * rowWords;
    if (idx >= total) { return; }
    let inner = idx % rowWords;
    let i = idx / rowWords;
    let srcRow = indices[i * params.z];
    out[idx] = src[srcRow * rowWords + inner];
}
