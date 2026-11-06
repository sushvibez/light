// BatchNorm2d eval over (N, C, H, W) with per-channel running stats.
// mode 0: out = (x - mean[c]) / sqrt(var[c] + eps) * gamma[c] + beta[c]
// mode 1: out = (x - mean[c]) / sqrt(var[c] + eps)            (x_hat)
// params: [C, HW, eps_bits, mode]
// params2: [total, 0, 0, 0]

@group(0) @binding(0) var<storage, read>       input: array<f32>;
@group(0) @binding(1) var<storage, read>       gamma: array<f32>;
@group(0) @binding(2) var<storage, read>       beta: array<f32>;
@group(0) @binding(3) var<storage, read>       mean: array<f32>;
@group(0) @binding(4) var<storage, read>       variance: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;
@group(0) @binding(6) var<uniform>             params: vec4<u32>;
@group(0) @binding(7) var<uniform>             params2: vec4<u32>;

const Y_STRIDE: u32 = 65535u * 256u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * Y_STRIDE;
    if (idx >= params2.x) { return; }
    let C = params.x;
    let HW = params.y;
    let eps = bitcast<f32>(params.z);
    let c = (idx / HW) % C;
    let xhat = (input[idx] - mean[c]) / sqrt(variance[c] + eps);
    if (params.w == 0u) {
        output[idx] = xhat * gamma[c] + beta[c];
    } else {
        output[idx] = xhat;
    }
}
