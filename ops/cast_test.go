package ops_test

import (
	"math"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// F8E4M3 round-trips: CPU conversion is the reference (dtype.F8FromF32);
// the CUDA cast kernels (__nv_cvt_* with SATFINITE) must agree bit-for-bit.
func TestCastF8E4M3(t *testing.T) {
	in := []float32{0, 1, -1, 0.5, 1.125, 448, -448, 1000, -1000, 1.0 / 512, 7.0 / 512, 1.0 / 1024,
		1.6 / 512, 1.0625, 1.1875, 3.3, 0.1, 100, 300, 460, -0.3, 2.5e-3, 17, 255.9, 1e-6}
	x, _ := tensor.New(in, []int{len(in)}, device.CPU())
	q, err := ops.Cast(x, dtype.F8E4M3)
	if err != nil {
		t.Fatal(err)
	}
	back, err := ops.Cast(q, dtype.F32)
	if err != nil {
		t.Fatal(err)
	}
	cpu := tensor.MustSlice[float32](back)
	for i, v := range in {
		want := dtype.F8FromF32(v).ToF32()
		if cpu[i] != want {
			t.Errorf("cpu cast %v: got %v want %v", v, cpu[i], want)
		}
	}
	gpu := cudaOrSkip(t)
	xg, _ := x.To(gpu)
	for _, mid := range []dtype.DType{dtype.F32, dtype.BF16, dtype.F16} {
		src := xg
		if mid != dtype.F32 {
			if src, err = ops.Cast(xg, mid); err != nil {
				t.Fatal(err)
			}
		}
		qg, err := ops.Cast(src, dtype.F8E4M3)
		if err != nil {
			t.Fatalf("%v→F8: %v", mid, err)
		}
		bg, err := ops.Cast(qg, mid)
		if err != nil {
			t.Fatalf("F8→%v: %v", mid, err)
		}
		if mid != dtype.F32 {
			if bg, err = ops.Cast(bg, dtype.F32); err != nil {
				t.Fatal(err)
			}
		}
		got := f32s(t, bg)
		// Reference: round through `mid` on the CPU, then F8.
		for i, v := range in {
			ref := v
			switch mid {
			case dtype.BF16:
				ref = dtype.BF16FromF32(v).ToF32()
			case dtype.F16:
				ref = dtype.HalfFromF32(v).ToF32()
			}
			want := dtype.F8FromF32(ref).ToF32()
			if got[i] != want && !(math.IsNaN(float64(got[i])) && math.IsNaN(float64(want))) {
				t.Errorf("cuda %v→F8→%v of %v: got %v want %v", mid, mid, v, got[i], want)
			}
		}
	}
}

// Every dtype must cast to and from F32 on CUDA exactly like the CPU kernels,
// and pairs without a direct kernel (e.g. I16 → U8) hop through F32.
func TestCastAllDTypesCUDAParity(t *testing.T) {
	gpu := cudaOrSkip(t)
	src := []float32{-3, 0, 1, 7, 250, 60000, 70000, -70000, 4000000000, 1.5, -2.5}
	cpu, err := tensor.New(src, []int{len(src)}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	dts := []dtype.DType{dtype.F16, dtype.BF16, dtype.F64, dtype.I8, dtype.U8, dtype.I16, dtype.U16, dtype.I32, dtype.U32, dtype.I64, dtype.U64}
	for _, dt := range dts {
		want, err := ops.Cast(cpu, dt)
		if err != nil {
			t.Fatal(err)
		}
		wantF, _ := ops.Cast(want, dtype.F32)
		g, _ := cpu.To(gpu)
		got, err := ops.Cast(g, dt)
		if err != nil {
			t.Fatalf("%v: %v", dt, err)
		}
		gotF, err := ops.Cast(got, dtype.F32)
		if err != nil {
			t.Fatalf("%v → F32: %v", dt, err)
		}
		a, b := f32s(t, wantF), f32s(t, gotF)
		for i := range a {
			// Float→int overflow is UB-ish on both sides; only compare in-range values.
			if dt.IsInt() && (src[i] < -1e9 || src[i] > 1e9) && dt != dtype.I64 && dt != dtype.U64 {
				continue
			}
			if a[i] != b[i] && !(a[i] != a[i] && b[i] != b[i]) {
				t.Errorf("%v[%d]: cpu %v cuda %v", dt, i, a[i], b[i])
			}
		}
	}
	// Two-hop pair on CUDA vs CPU.
	i16, _ := ops.Cast(cpu, dtype.I16)
	u8c, _ := ops.Cast(i16, dtype.U8)
	i16g, _ := i16.To(gpu)
	u8g, err := ops.Cast(i16g, dtype.U8)
	if err != nil {
		t.Fatal(err)
	}
	if u8g.DType() != dtype.U8 {
		t.Fatalf("dtype %v", u8g.DType())
	}
	expectRow(t, "i16→u8 hop", f32s(t, u8g), f32s(t, u8c), 0)
}
