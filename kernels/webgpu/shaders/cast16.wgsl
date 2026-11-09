// Casts between f32 and the 16-bit float types, which are stored two per
// u32 word (element 2i in the low half, 2i+1 in the high half).
// params: [mode, n (elements), 0, 0]
// mode 0: f32 -> bf16 (round to nearest even; NaN stays NaN)
//      1: bf16 -> f32 (exact)
//      2: f32 -> f16 (round to nearest even, bit for bit dtype.HalfFromF32;
//         pack2x16float's rounding is implementation-defined)
//      3: f16 -> f32 (unpack2x16float)
// Narrowing runs one invocation per output word, widening one per element.

@group(0) @binding(0) var<storage, read>       input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform>             params: vec4<u32>;

const Y_STRIDE: u32 = 65535u * 256u;

fn bf16_bits(x: u32) -> u32 {
    if ((x & 0x7fffffffu) > 0x7f800000u) { return (x >> 16u) | 0x40u; } // quiet NaN
    let lsb = (x >> 16u) & 1u;
    return (x + 0x7fffu + lsb) >> 16u;
}

fn f16_bits(x: u32) -> u32 {
    let sign = (x >> 16u) & 0x8000u;
    let e = (x >> 23u) & 0xffu;
    let m = x & 0x7fffffu;
    if (e == 0xffu) {
        if (m == 0u) { return sign | 0x7c00u; }
        return sign | 0x7c00u | (m >> 13u) | 0x200u; // quiet NaN
    }
    if (e < 103u) { return sign; }           // below 2^-24 (and F32 subnormals)
    if (e > 142u) { return sign | 0x7c00u; } // above 2^15: inf
    if (e < 113u) {                          // F16 subnormal
        let mm = m | 0x800000u;
        let shift = 126u - e;
        var h = mm >> shift;
        let rem = mm & ((1u << shift) - 1u);
        let half = 1u << (shift - 1u);
        if (rem > half || (rem == half && (h & 1u) == 1u)) { h += 1u; }
        return sign | h;
    }
    var v = ((e - 112u) << 10u) | (m >> 13u);
    let rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (v & 1u) == 1u)) { v += 1u; } // a carry bumps the exponent (to inf at most)
    return sign | v;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * Y_STRIDE;
    let n = params.y;
    switch params.x {
        case 0u, 2u: { // narrow: output word idx holds elements 2idx, 2idx+1
            let e = 2u * idx;
            if (e >= n) { return; }
            let a = input[e];
            var b = 0u;
            if (e + 1u < n) { b = input[e + 1u]; }
            if (params.x == 0u) {
                output[idx] = bf16_bits(a) | (bf16_bits(b) << 16u);
            } else {
                output[idx] = f16_bits(a) | (f16_bits(b) << 16u);
            }
        }
        case 1u, 3u: { // widen: one f32 per element
            if (idx >= n) { return; }
            let w = input[idx / 2u];
            if (params.x == 1u) {
                let h = select(w & 0xffffu, w >> 16u, (idx & 1u) == 1u);
                output[idx] = h << 16u;
            } else {
                let v = unpack2x16float(w);
                output[idx] = bitcast<u32>(select(v.x, v.y, (idx & 1u) == 1u));
            }
        }
        default: {}
    }
}
