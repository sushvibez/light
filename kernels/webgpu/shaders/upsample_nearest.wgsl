// Nearest-neighbour upsampling of (N, C, H, W) by integer factors:
// out[nc, oy, ox] = in[nc, oy / sh, ox / sw].
// params: [OH, OW, H, W]; one invocation per output element.

@group(0) @binding(0) var<storage, read>       input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform>             params: vec4<u32>;

const Y_STRIDE: u32 = 65535u * 256u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * Y_STRIDE;
    if (idx >= arrayLength(&output)) { return; }
    let OH = params.x;
    let OW = params.y;
    let H = params.z;
    let W = params.w;
    let plane = OH * OW;
    let nc = idx / plane;
    let rem = idx % plane;
    let oy = rem / OW;
    let ox = rem % OW;
    output[idx] = input[nc * H * W + (oy / (OH / H)) * W + ox / (OW / W)];
}
