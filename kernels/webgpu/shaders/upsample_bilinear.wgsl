// Bilinear resampling of (N, C, H, W) to (N, C, OH, OW)
// (F.interpolate(mode="bilinear")): with align_corners the source
// coordinate is o·(in-1)/(out-1), otherwise max((o+0.5)·in/out - 0.5, 0).
// params: [OH, OW, H, W], [align, 0, 0, 0]; one invocation per output element.

@group(0) @binding(0) var<storage, read>       input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform>             params: array<vec4<u32>, 2>;

const Y_STRIDE: u32 = 65535u * 256u;

struct Src { i0: u32, i1: u32, l: f32 }

fn source(o: u32, n: u32, out: u32, align: bool) -> Src {
    var s: f32 = 0.0;
    if (align) {
        if (out > 1u) {
            s = f32(o) * f32(n - 1u) / f32(out - 1u);
        }
    } else {
        s = max((f32(o) + 0.5) * f32(n) / f32(out) - 0.5, 0.0);
    }
    let i0 = min(u32(floor(s)), n - 1u);
    let i1 = min(i0 + 1u, n - 1u);
    return Src(i0, i1, s - f32(i0));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + gid.y * Y_STRIDE;
    if (idx >= arrayLength(&output)) { return; }
    let OH = params[0].x;
    let OW = params[0].y;
    let H = params[0].z;
    let W = params[0].w;
    let align = params[1].x != 0u;
    let plane = OH * OW;
    let nc = idx / plane;
    let rem = idx % plane;
    let sy = source(rem / OW, H, OH, align);
    let sx = source(rem % OW, W, OW, align);
    let base = nc * H * W;
    let a = input[base + sy.i0 * W + sx.i0];
    let b = input[base + sy.i0 * W + sx.i1];
    let c = input[base + sy.i1 * W + sx.i0];
    let d = input[base + sy.i1 * W + sx.i1];
    output[idx] = (1.0 - sy.l) * ((1.0 - sx.l) * a + sx.l * b) + sy.l * ((1.0 - sx.l) * c + sx.l * d);
}
