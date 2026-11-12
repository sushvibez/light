package cpu_test

import (
	"encoding/binary"
	"math"
	"math/rand/v2"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// The typed float casts match the element-wise conversions bit for bit,
// including NaN, ±Inf, subnormals and values that round across the 16-bit
// formats' ranges.
func TestCastFloatFastBitExact(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	n := 300_001
	src := make([]float32, n)
	for i := range src {
		src[i] = float32(rng.NormFloat64() * math.Pow(10, float64(rng.IntN(20)-10)))
	}
	copy(src, []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1)), 0, -0.0, 1e-40, 70000, -65520, 3.4e38})
	x, err := tensor.New(src, []int{n}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	for _, dt := range []dtype.DType{dtype.BF16, dtype.F16} {
		h, err := ops.Cast(x, dt)
		if err != nil {
			t.Fatal(err)
		}
		raw := h.Storage().CPUBytes()
		bits := make([]uint16, n)
		for i := range bits {
			bits[i] = binary.LittleEndian.Uint16(raw[2*i:])
		}
		back, err := ops.Cast(h, dtype.F32)
		if err != nil {
			t.Fatal(err)
		}
		got := tensor.MustSlice[float32](back)
		for i, v := range src {
			var want uint16
			var wantF float32
			if dt == dtype.BF16 {
				want = uint16(dtype.BF16FromF32(v))
				wantF = dtype.BFloat16(want).ToF32()
			} else {
				want = uint16(dtype.HalfFromF32(v))
				wantF = dtype.Half(want).ToF32()
			}
			if bits[i] != want {
				t.Fatalf("%v: element %d (%g): bits %#04x, want %#04x", dt, i, v, bits[i], want)
			}
			if math.Float32bits(got[i]) != math.Float32bits(wantF) && !(math.IsNaN(float64(got[i])) && math.IsNaN(float64(wantF))) {
				t.Fatalf("%v→F32: element %d: %g, want %g", dt, i, got[i], wantF)
			}
		}
	}
}

func BenchmarkCastF32ToBF16(b *testing.B) {
	n := 1 << 24
	x, _ := tensor.New(make([]float32, n), []int{n}, device.CPU())
	b.SetBytes(int64(4 * n))
	for b.Loop() {
		if _, err := ops.Cast(x, dtype.BF16); err != nil {
			b.Fatal(err)
		}
	}
}
