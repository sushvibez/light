// GGUF block decoding shared by gguf_dequant and gguf_gemv. The launcher
// prepends `const QTYPE: u32 = …; const BE: u32 = …; const BB: u32 = …;`
// (format id, elements and bytes per block). Weights are raw block bytes
// bound as array<u32>; blocks need not be word-aligned (Q8_0 is 34 bytes,
// Q6_K 210), so every field is read bytewise. Decoders mirror light/quant.

@group(0) @binding(0) var<storage, read> w: array<u32>;

fn byte_at(b: u32) -> u32 { return (w[b >> 2u] >> ((b & 3u) * 8u)) & 0xFFu; }
fn f16_at(b: u32) -> f32 { return unpack2x16float(byte_at(b) | (byte_at(b + 1u) << 8u)).x; }
fn i8_at(b: u32) -> f32 { let v = i32(byte_at(b)); return f32(select(v, v - 256, v > 127)); }

// Q4_K's packed 6-bit (scale, min) pair j of the 12 bytes at sb.
fn scale_min_k4(j: u32, sb: u32) -> vec2<f32> {
    if (j < 4u) {
        return vec2<f32>(f32(byte_at(sb + j) & 63u), f32(byte_at(sb + j + 4u) & 63u));
    }
    let sc = (byte_at(sb + j + 4u) & 15u) | ((byte_at(sb + j - 4u) >> 6u) << 4u);
    let m = (byte_at(sb + j + 4u) >> 4u) | ((byte_at(sb + j) >> 6u) << 4u);
    return vec2<f32>(f32(sc), f32(m));
}

// deq returns element i (< BE) of the block starting at byte base.
fn deq(base: u32, i: u32) -> f32 {
    switch QTYPE {
        case 8u: { // Q8_0: f16 d, 32 × i8
            return f16_at(base) * i8_at(base + 2u + i);
        }
        case 2u: { // Q4_0: f16 d, 16 bytes of nibbles (low: 0..15, high: 16..31), offset 8
            let q = byte_at(base + 2u + (i & 15u));
            let nib = select(q & 15u, q >> 4u, i >= 16u);
            return f16_at(base) * f32(i32(nib) - 8);
        }
        case 6u: { // Q5_0: f16 d, u32 qh (bit i = fifth bit of element i), 16 bytes of nibbles, offset 16
            let q = byte_at(base + 6u + (i & 15u));
            let nib = select(q & 15u, q >> 4u, i >= 16u);
            let hb = (byte_at(base + 2u + (i >> 3u)) >> (i & 7u)) & 1u;
            return f16_at(base) * f32(i32(nib | (hb << 4u)) - 16);
        }
        case 12u: { // Q4_K: f16 d, f16 dmin, 12 scale bytes, 128 bytes of nibbles
            let p = i / 64u;
            let r = i % 64u;
            let hi = r >= 32u;
            let l = r % 32u;
            let smin = scale_min_k4(2u * p + select(0u, 1u, hi), base + 4u);
            let q = byte_at(base + 16u + p * 32u + l);
            let nib = select(q & 15u, q >> 4u, hi);
            return f16_at(base) * smin.x * f32(nib) - f16_at(base + 2u) * smin.y;
        }
        case 14u: { // Q6_K: ql[128], qh[64], i8 scales[16], f16 d
            let half = i / 128u;
            let r = i % 128u;
            let quarter = r / 32u;
            let l = r % 32u;
            let ql = base + half * 64u;
            let qh = byte_at(base + 128u + half * 32u + l);
            let lo = byte_at(ql + l + select(0u, 32u, (quarter & 1u) == 1u));
            let nib = select(lo & 15u, lo >> 4u, quarter >= 2u);
            let raw = nib | (((qh >> (2u * quarter)) & 3u) << 4u);
            let s = i8_at(base + 192u + half * 8u + 2u * quarter + (l >> 4u));
            return f16_at(base + 208u) * s * f32(i32(raw) - 32);
        }
        default: { return 0.0; }
    }
}

// row_elem is element k of weight row n (K elements per row).
fn row_elem(n: u32, K: u32, k: u32) -> f32 {
    return deq(n * (K / BE) * BB + (k / BE) * BB, k % BE);
}
