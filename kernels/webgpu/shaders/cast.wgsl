// dtype casts on 32-bit words.
// params: [mode, len, 0, 0]
// mode: 0=f32->i32  1=i32->f32  2=f32->i64  3=i64->f32  4=i32->i64  5=i64->i32
//       6=f32->u32  7=u32->f32
// 64-bit integers are two little-endian words (low, high); WGSL has no i64,
// so f32->i64 sign-extends an i32 conversion and i64->f32 combines the halves.

@group(0) @binding(0) var<storage, read>       input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform>             params: vec4<u32>;

const Y_STRIDE: u32 = 65535u * 256u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * Y_STRIDE;
    if (idx >= params.y) { return; }
    switch params.x {
        case 0u: {
            output[idx] = bitcast<u32>(i32(bitcast<f32>(input[idx])));
        }
        case 1u: {
            output[idx] = bitcast<u32>(f32(bitcast<i32>(input[idx])));
        }
        case 2u: {
            let v = i32(bitcast<f32>(input[idx]));
            output[2u * idx] = bitcast<u32>(v);
            output[2u * idx + 1u] = select(0u, 0xffffffffu, v < 0);
        }
        case 3u: {
            let lo = input[2u * idx];
            let hi = bitcast<i32>(input[2u * idx + 1u]);
            var v: f32;
            if (hi == 0) {
                v = f32(lo);
            } else if (hi == -1) {
                v = f32(bitcast<i32>(lo)); // small negative: exact in i32
            } else {
                v = f32(hi) * 4294967296.0 + f32(lo);
            }
            output[idx] = bitcast<u32>(v);
        }
        case 4u: {
            let v = bitcast<i32>(input[idx]);
            output[2u * idx] = bitcast<u32>(v);
            output[2u * idx + 1u] = select(0u, 0xffffffffu, v < 0);
        }
        case 5u: {
            output[idx] = input[2u * idx];
        }
        case 6u: {
            output[idx] = u32(bitcast<f32>(input[idx]));
        }
        case 7u: {
            output[idx] = bitcast<u32>(f32(input[idx]));
        }
        default: {}
    }
}
