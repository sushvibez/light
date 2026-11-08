package cpu

import (
	"encoding/binary"
	"math"
	"math/rand/v2"
	"testing"

	"light/dtype"
	"light/internal/simd"
)

// randomRow is K elements of random GGUF blocks of dt with finite f16
// scale fields.
func randomRow(rng *rand.Rand, dt dtype.DType, K int) []byte {
	dec := ggufDecoders[dt]
	raw := make([]byte, K/dec.elems*dec.bytes)
	for i := range raw {
		raw[i] = byte(rng.Uint32())
	}
	f := func(off int, v float32) { binary.LittleEndian.PutUint16(raw[off:], uint16(dtype.HalfFromF32(v))) }
	for b := 0; b < len(raw)/dec.bytes; b++ {
		o := b * dec.bytes
		switch dt {
		case dtype.Q4_K:
			f(o, 0.01*rng.Float32())
			f(o+2, 0.01*rng.Float32())
		case dtype.Q6_K:
			f(o+208, 0.01*rng.Float32()-0.005)
		default:
			f(o, 0.1*rng.Float32()-0.05)
		}
	}
	return raw
}

// The AVX2 row decoders produce exactly the Go decoders' floats.
func TestAVX2DecodersExact(t *testing.T) {
	if !decodeAVX2 {
		t.Skip("no AVX2")
	}
	rng := rand.New(rand.NewPCG(11, 12))
	const K = 1024
	for dt := range ggufAVX2 {
		dec := ggufDecoders[dt]
		for range 20 {
			raw := randomRow(rng, dt, K)
			want, got, ds := make([]float32, K), make([]float32, K), make([]float32, K/32)
			for b := 0; b < K/dec.elems; b++ {
				dec.decode(raw[b*dec.bytes:], want[b*dec.elems:])
			}
			if !decodeRowAVX2(dt, raw, K, got, ds) {
				t.Fatalf("%v: no AVX2 decoder", dt)
			}
			for i := range want {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("%v [%d] = %g, Go decoder %g", dt, i, got[i], want[i])
				}
			}
		}
	}
	// GPTQ / AWQ int4 rows, group 32 and 128.
	for _, G := range []int{32, 128} {
		codes := make([]uint32, K/8)
		s, z := make([]float32, K/G), make([]float32, K/G)
		for i := range codes {
			codes[i] = rng.Uint32()
		}
		for i := range s {
			s[i], z[i] = 0.01*rng.Float32(), 8*rng.Float32()
		}
		got := make([]float32, K)
		int4RowAVX2(&codes[0], &s[0], &z[0], &got[0], K/G, G/8)
		for k := range K {
			g := k / G
			want := s[g] * (float32((codes[k/8]>>(4*(k%8)))&15) - z[g])
			if math.Float32bits(got[k]) != math.Float32bits(want) {
				t.Fatalf("int4 G=%d [%d] = %g, want %g", G, k, got[k], want)
			}
		}
	}
}

// Row decode cost, Go vs AVX2, against one SIMD dot over the row.
func BenchmarkRowDecode(b *testing.B) {
	const K = 4864 / 256 * 256
	rng := rand.New(rand.NewPCG(1, 1))
	dst, ds, x := make([]float32, K), make([]float32, K/32), make([]float32, K)
	for _, dt := range []dtype.DType{dtype.Q8_0, dtype.Q4_0, dtype.Q5_0, dtype.Q4_K, dtype.Q6_K} {
		dec := ggufDecoders[dt]
		raw := randomRow(rng, dt, K)
		b.Run(dt.String()+"/go", func(b *testing.B) {
			for b.Loop() {
				for blk := 0; blk < K/dec.elems; blk++ {
					dec.decode(raw[blk*dec.bytes:], dst[blk*dec.elems:])
				}
			}
		})
		if decodeAVX2 {
			b.Run(dt.String()+"/avx2", func(b *testing.B) {
				for b.Loop() {
					decodeRowAVX2(dt, raw, K, dst, ds)
				}
			})
		}
	}
	b.Run("dot", func(b *testing.B) {
		for b.Loop() {
			_ = simd.Dot(dst, x)
		}
	})
}
