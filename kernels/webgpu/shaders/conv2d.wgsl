// Direct 2-D convolution, one thread per output element.
// params0: [N, Cin, Cout, H]
// params1: [W, kH, kW, sH]
// params2: [sW, pH, pW, Hout]
// params3: [Wout, hasBias, 0, 0]

@group(0) @binding(0) var<storage, read>       input: array<f32>;
@group(0) @binding(1) var<storage, read>       weight: array<f32>;
@group(0) @binding(2) var<storage, read>       bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform>             params0: vec4<u32>;
@group(0) @binding(5) var<uniform>             params1: vec4<u32>;
@group(0) @binding(6) var<uniform>             params2: vec4<u32>;
@group(0) @binding(7) var<uniform>             params3: vec4<u32>;

const Y_STRIDE: u32 = 65535u * 256u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * Y_STRIDE;
    let N = params0.x;
    let Cin = params0.y;
    let Cout = params0.z;
    let H = params0.w;
    let W = params1.x;
    let kH = params1.y;
    let kW = params1.z;
    let sH = params1.w;
    let sW = params2.x;
    let pH = params2.y;
    let pW = params2.z;
    let Hout = params2.w;
    let Wout = params3.x;
    let hasBias = params3.y;

    let total = N * Cout * Hout * Wout;
    if (idx >= total) { return; }

    let ow = idx % Wout;
    let t1 = idx / Wout;
    let oh = t1 % Hout;
    let t2 = t1 / Hout;
    let oc = t2 % Cout;
    let n = t2 / Cout;

    var sum: f32 = 0.0;
    for (var ic: u32 = 0u; ic < Cin; ic = ic + 1u) {
        for (var kh: u32 = 0u; kh < kH; kh = kh + 1u) {
            let ih = i32(oh * sH + kh) - i32(pH);
            if (ih < 0 || ih >= i32(H)) { continue; }
            for (var kw: u32 = 0u; kw < kW; kw = kw + 1u) {
                let iw = i32(ow * sW + kw) - i32(pW);
                if (iw < 0 || iw >= i32(W)) { continue; }
                let inIdx = ((n * Cin + ic) * H + u32(ih)) * W + u32(iw);
                let wIdx = ((oc * Cin + ic) * kH + kh) * kW + kw;
                sum = sum + input[inIdx] * weight[wIdx];
            }
        }
    }
    if (hasBias != 0u) {
        sum = sum + bias[oc];
    }
    output[idx] = sum;
}
