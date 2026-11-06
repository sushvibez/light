// Max / average pooling over (N, C, H, W), one thread per output element.
// Padding cells are skipped; avg divides by the count of in-bounds cells
// (matching kernels/cpu).
// params0: [N, C, H, W]
// params1: [Hout, Wout, kH, kW]
// params2: [sH, sW, pH, pW]
// params3: [mode, 0, 0, 0]   mode: 0=max 1=avg

@group(0) @binding(0) var<storage, read>       input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform>             params0: vec4<u32>;
@group(0) @binding(3) var<uniform>             params1: vec4<u32>;
@group(0) @binding(4) var<uniform>             params2: vec4<u32>;
@group(0) @binding(5) var<uniform>             params3: vec4<u32>;

const Y_STRIDE: u32 = 65535u * 256u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * Y_STRIDE;
    let N = params0.x;
    let C = params0.y;
    let H = params0.z;
    let W = params0.w;
    let Hout = params1.x;
    let Wout = params1.y;
    let kH = params1.z;
    let kW = params1.w;
    let sH = params2.x;
    let sW = params2.y;
    let pH = params2.z;
    let pW = params2.w;
    let avg = params3.x == 1u;

    let total = N * C * Hout * Wout;
    if (idx >= total) { return; }

    let ow = idx % Wout;
    let t1 = idx / Wout;
    let oh = t1 % Hout;
    let nc = t1 / Hout;
    let base = nc * H * W;

    var acc: f32 = 0.0;
    var count: u32 = 0u;
    var first = true;
    for (var kh: u32 = 0u; kh < kH; kh = kh + 1u) {
        let ih = i32(oh * sH + kh) - i32(pH);
        if (ih < 0 || ih >= i32(H)) { continue; }
        for (var kw: u32 = 0u; kw < kW; kw = kw + 1u) {
            let iw = i32(ow * sW + kw) - i32(pW);
            if (iw < 0 || iw >= i32(W)) { continue; }
            let v = input[base + u32(ih) * W + u32(iw)];
            if (avg) {
                acc = acc + v;
                count = count + 1u;
            } else if (first || v > acc) {
                acc = v;
                first = false;
            }
        }
    }
    if (avg && count > 0u) {
        acc = acc / f32(count);
    }
    output[idx] = acc;
}
