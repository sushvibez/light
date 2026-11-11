// Unary elementwise op on contiguous f32.
// params: [op, len, 0, 0]
// op: 0=relu 1=sigmoid 2=tanh 3=neg 4=exp 5=log 6=sqrt 7=gelu(tanh approx)
//     8=silu 9=abs 10=sin 11=cos 12=tan 13=floor 14=ceil 15=round(half away from zero)
//     16=softplus

@group(0) @binding(0) var<storage, read>       input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform>             params: vec4<u32>;

const Y_STRIDE: u32 = 65535u * 256u;
const SQRT_2_OVER_PI: f32 = 0.7978845608;
const GELU_COEFF: f32 = 0.044715;

// tanh with the argument clamped to ±15, where tanh is ±1 to f32 precision.
// The builtin is not safe everywhere: on D3D12 (HLSL tanh) some drivers
// return NaN for large inputs (seen on Intel graphics: GPT-2's GELU).
// erf via Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7; WGSL has no erf).
fn erf_f32(x: f32) -> f32 {
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
    return sign(x) * y;
}

fn safe_tanh(x: f32) -> f32 {
    return tanh(clamp(x, -15.0, 15.0));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * Y_STRIDE;
    if (idx >= params.y) { return; }
    let x = input[idx];
    switch params.x {
        case 0u:  { out[idx] = max(x, 0.0); }
        case 1u:  { out[idx] = 1.0 / (1.0 + exp(-x)); }
        case 2u:  { out[idx] = safe_tanh(x); }
        case 3u:  { out[idx] = -x; }
        case 4u:  { out[idx] = exp(x); }
        case 5u:  { out[idx] = log(x); }
        case 6u:  { out[idx] = sqrt(x); }
        case 7u:  {
            let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x * x * x);
            out[idx] = 0.5 * x * (1.0 + safe_tanh(inner));
        }
        case 8u:  { out[idx] = x / (1.0 + exp(-x)); }
        case 9u:  { out[idx] = abs(x); }
        case 10u: { out[idx] = sin(x); }
        case 11u: { out[idx] = cos(x); }
        case 12u: { out[idx] = tan(x); }
        case 13u: { out[idx] = floor(x); }
        case 14u: { out[idx] = ceil(x); }
        case 15u: { out[idx] = sign(x) * floor(abs(x) + 0.5); }
        case 16u: { // softplus with torch's threshold 20; exp(x) where log1p(e) ~ e
            if (x > 20.0) {
                out[idx] = x;
            } else if (x < -15.0) {
                out[idx] = exp(x);
            } else {
                out[idx] = log(1.0 + exp(x));
            }
        }
        case 18u: { out[idx] = min(max(x, bitcast<f32>(params.z)), bitcast<f32>(params.w)); } // clamp(lo, hi)
        case 17u: { out[idx] = 0.5 * x * (1.0 + erf_f32(x * 0.70710678118654752)); } // exact GELU
        default: {}
    }
}
