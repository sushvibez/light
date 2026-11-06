// Broadcast / strided binary elementwise op on f32.
// shp: [dims[rank], aStrides[rank], bStrides[rank]] (i32, element units,
//       broadcast axes carry stride 0)
// params:  [op, len, rank, 0]
// params2: [aOffset, bOffset, 0, 0]  (element offsets into a / b)

@group(0) @binding(0) var<storage, read>       a: array<f32>;
@group(0) @binding(1) var<storage, read>       b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read>       shp: array<i32>;
@group(0) @binding(4) var<uniform>             params: vec4<u32>;
@group(0) @binding(5) var<uniform>             params2: vec4<u32>;

const Y_STRIDE: u32 = 65535u * 256u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * Y_STRIDE;
    if (idx >= params.y) { return; }
    let rank = params.z;
    var rem = i32(idx);
    var aOff = i32(params2.x);
    var bOff = i32(params2.y);
    // Walk axes from the innermost outwards.
    for (var k: u32 = 0u; k < rank; k = k + 1u) {
        let ax = rank - 1u - k;
        let d = shp[ax];
        let c = rem % d;
        rem = rem / d;
        aOff = aOff + c * shp[rank + ax];
        bOff = bOff + c * shp[2u * rank + ax];
    }
    let x = a[u32(aOff)];
    let y = b[u32(bOff)];
    switch params.x {
        case 0u: { out[idx] = x + y; }
        case 1u: { out[idx] = x - y; }
        case 2u: { out[idx] = x * y; }
        case 3u: { out[idx] = x / y; }
        case 4u: { out[idx] = pow(x, y); }
        default: {}
    }
}
