package cpu

import (
	"math"

	"light/device"
	"light/dtype"
	"light/internal/registry"
)

func init() {
	registry.Register(registry.Key{Op: "kv_ring_write", Dev: device.KindCPU, DType: dtype.F32}, kvRingWriteCPU)
	registry.Register(registry.Key{Op: "attn_decode", Dev: device.KindCPU, DType: dtype.F32}, attnDecodeCPU)
}

// kvRingWriteCPU: ops.KVRingWrite. Input src (B, H, T, D) with element
// strides sb / sh / st (last axis contiguous); output the ring (B, H, Cap, D).
func kvRingWriteCPU(ctx *registry.Context) error {
	src, ring := ctx.Inputs[0], ctx.Output
	s := asSlice[float32](src)[src.Offset():]
	r := asSlice[float32](ring)
	B, H, T, D, Cap := src.Dim(0), src.Dim(1), src.Dim(2), src.Dim(3), ring.Dim(2)
	slot0, sb, sh, st := ctx.Attrs["slot0"].(int), ctx.Attrs["sb"].(int), ctx.Attrs["sh"].(int), ctx.Attrs["st"].(int)
	for b := range B {
		for h := range H {
			for t := range T {
				slot := (slot0 + t) % Cap
				copy(r[((b*H+h)*Cap+slot)*D:][:D], s[b*sb+h*sh+t*st:][:D])
			}
		}
	}
	return nil
}

// attnDecodeCPU: ops.AttnDecode. Inputs q (B, H, 1, D) with strides qb / qh,
// rings k, v (B, H, Cap, D); attends slots [0, n).
func attnDecodeCPU(ctx *registry.Context) error {
	q, k, v, out := ctx.Inputs[0], ctx.Inputs[1], ctx.Inputs[2], ctx.Output
	qs := asSlice[float32](q)[q.Offset():]
	ks, vs, os := asSlice[float32](k), asSlice[float32](v), asSlice[float32](out)
	B, H, D, Cap := q.Dim(0), q.Dim(1), q.Dim(3), k.Dim(2)
	n, scale, qb, qh := ctx.Attrs["n"].(int), ctx.Attrs["scale"].(float32), ctx.Attrs["qb"].(int), ctx.Attrs["qh"].(int)
	p := make([]float32, n)
	for b := range B {
		for h := range H {
			qv := qs[b*qb+h*qh:][:D]
			base := (b*H + h) * Cap * D
			m := float32(math.Inf(-1))
			for j := range n {
				kj := ks[base+j*D:][:D]
				var dot float32
				for d, x := range qv {
					dot += x * kj[d]
				}
				p[j] = dot * scale
				m = max(m, p[j])
			}
			var l float32
			for j := range p {
				p[j] = float32(math.Exp(float64(p[j] - m)))
				l += p[j]
			}
			o := os[(b*H+h)*D:][:D]
			for j, pj := range p {
				w := pj / l
				vj := vs[base+j*D:][:D]
				for d := range o {
					o[d] += w * vj[d]
				}
			}
		}
	}
	return nil
}
