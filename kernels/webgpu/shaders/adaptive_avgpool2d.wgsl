// Adaptive average pool over (N, C, H, W) -> (N, C, OH, OW), PyTorch window
// convention: start = floor(o*L/OL), end = ceil((o+1)*L/OL).
// params0: [N, C, H, W]
// params1: [OH, OW, 0, 0]

@group(0) @binding(0) var<storage, read>       input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform>             params0: vec4<u32>;
@group(0) @binding(3) var<uniform>             params1: vec4<u32>;

const Y_STRIDE: u32 = 65535u * 256u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * Y_STRIDE;
    let N = params0.x;
    let C = params0.y;
    let H = params0.z;
    let W = params0.w;
    let OH = params1.x;
    let OW = params1.y;
    let total = N * C * OH * OW;
    if (idx >= total) { return; }

    let ow = idx % OW;
    let t1 = idx / OW;
    let oh = t1 % OH;
    let nc = t1 / OH;
    let base = nc * H * W;

    let hStart = (oh * H) / OH;
    let hEnd = min(((oh + 1u) * H + OH - 1u) / OH, H);
    let wStart = (ow * W) / OW;
    let wEnd = min(((ow + 1u) * W + OW - 1u) / OW, W);

    var sum: f32 = 0.0;
    var count: u32 = 0u;
    for (var ih = hStart; ih < hEnd; ih = ih + 1u) {
        for (var iw = wStart; iw < wEnd; iw = iw + 1u) {
            sum = sum + input[base + ih * W + iw];
            count = count + 1u;
        }
    }
    if (count > 0u) {
        output[idx] = sum / f32(count);
    } else {
        output[idx] = 0.0;
    }
}
