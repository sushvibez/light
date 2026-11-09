// Gated delta rule, recurrent form (transformers' torch_recurrent_gated_delta_rule):
// per (batch, head), state S (dk x dv), for each token t:
//   S <- exp(g_t) S;  delta = beta_t (v_t - S^T k_t);  S <- S + k_t (x) delta;  o_t = S^T q_t
// with q scaled by 1/sqrt(dk) and q, k optionally L2-normalised. One
// workgroup per (b, h); thread j owns column j of S in a private array
// (dk <= 128, dv <= 128); q and k of the current token are staged in
// shared memory.
// params: [B*H, T, H, dk | dv << 8 | l2 << 16]

@group(0) @binding(0) var<storage, read>       q: array<f32>;
@group(0) @binding(1) var<storage, read>       k: array<f32>;
@group(0) @binding(2) var<storage, read>       v: array<f32>;
@group(0) @binding(3) var<storage, read>       g: array<f32>;
@group(0) @binding(4) var<storage, read>       beta: array<f32>;
@group(0) @binding(5) var<storage, read>       st: array<f32>;
@group(0) @binding(6) var<storage, read_write> fin: array<f32>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform>             params: vec4<u32>;

var<workgroup> qs: array<f32, 128>;
var<workgroup> ks: array<f32, 128>;
var<workgroup> scale: vec2<f32>;

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let bh = wid.x;
    let T = params.y;
    let H = params.z;
    let dk = params.w & 0xFFu;
    let dv = (params.w >> 8u) & 0xFFu;
    let l2 = (params.w >> 16u) != 0u;
    let b = bh / H;
    let h = bh % H;
    let j = lid.x;
    let live = j < dv;
    let sBase = bh * dk * dv;
    var S: array<f32, 128>;
    for (var i = 0u; i < dk; i++) {
        if (live) { S[i] = st[sBase + i * dv + j]; } else { S[i] = 0.0; }
    }
    for (var t = 0u; t < T; t++) {
        let gi = (b * T + t) * H + h;
        let qkBase = gi * dk;
        if (j < dk) {
            qs[j] = q[qkBase + j];
            ks[j] = k[qkBase + j];
        }
        workgroupBarrier();
        if (j == 0u) {
            var qq = 0.0;
            var kk = 0.0;
            for (var i = 0u; i < dk; i++) {
                qq += qs[i] * qs[i];
                kk += ks[i] * ks[i];
            }
            var sq = 1.0 / sqrt(f32(dk));
            var sk = 1.0;
            if (l2) {
                sq = sq / sqrt(qq + 1e-6);
                sk = 1.0 / sqrt(kk + 1e-6);
            }
            scale = vec2<f32>(sq, sk);
        }
        workgroupBarrier();
        if (live) {
            let decay = exp(g[gi]);
            let bt = beta[gi];
            var mem = 0.0;
            for (var i = 0u; i < dk; i++) {
                S[i] = S[i] * decay;
                mem += S[i] * ks[i] * scale.y;
            }
            let delta = (v[gi * dv + j] - mem) * bt;
            var o = 0.0;
            for (var i = 0u; i < dk; i++) {
                S[i] += ks[i] * scale.y * delta;
                o += S[i] * qs[i] * scale.x;
            }
            out[gi * dv + j] = o;
        }
        workgroupBarrier(); // qs / ks are rewritten for the next token
    }
    if (live) {
        for (var i = 0u; i < dk; i++) {
            fin[sBase + i * dv + j] = S[i];
        }
    }
}
