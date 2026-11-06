package ops_test

import (
	"math"
	"testing"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

func f32Close(t *testing.T, name string, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len %d != %d", name, len(got), len(want))
	}
	for i := range want {
		if math.Abs(float64(got[i]-want[i])) > tol {
			t.Errorf("%s[%d] = %v, want %v", name, i, got[i], want[i])
		}
	}
}

func seqTensor(t *testing.T, dims []int, scale float32) *tensor.Tensor {
	t.Helper()
	n := 1
	for _, d := range dims {
		n *= d
	}
	data := make([]float32, n)
	for i := range data {
		data[i] = float32((i*7)%11-5) * scale
	}
	x, err := tensor.New(data, dims, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	return x
}

func TestEinsumMatMul(t *testing.T) {
	a := seqTensor(t, []int{3, 4}, 0.5)
	b := seqTensor(t, []int{4, 5}, 0.25)
	got, err := ops.Einsum("ij,jk->ik", a, b)
	if err != nil {
		t.Fatal(err)
	}
	want, _ := ops.MatMul(a, b)
	if got.Rank() != 2 || got.Dim(0) != 3 || got.Dim(1) != 5 {
		t.Fatalf("shape %v", got.Shape())
	}
	f32Close(t, "ij,jk->ik", tensor.MustSlice[float32](got), tensor.MustSlice[float32](want), 1e-5)
}

func TestEinsumMatMulTransposedOperand(t *testing.T) {
	// ik,jk->ij == A @ Bᵀ
	a := seqTensor(t, []int{3, 4}, 0.5)
	b := seqTensor(t, []int{5, 4}, 0.25)
	got, err := ops.Einsum("ik,jk->ij", a, b)
	if err != nil {
		t.Fatal(err)
	}
	bT, _ := b.Transpose(0, 1)
	bTc, _ := bT.Contiguous()
	want, _ := ops.MatMul(a, bTc)
	f32Close(t, "ik,jk->ij", tensor.MustSlice[float32](got), tensor.MustSlice[float32](want), 1e-5)
}

func TestEinsumBatchedMatMul(t *testing.T) {
	a := seqTensor(t, []int{2, 3, 4}, 0.5)
	b := seqTensor(t, []int{2, 4, 5}, 0.25)
	got, err := ops.Einsum("bij,bjk->bik", a, b)
	if err != nil {
		t.Fatal(err)
	}
	want, _ := ops.MatMul(a, b)
	f32Close(t, "bij,bjk->bik", tensor.MustSlice[float32](got), tensor.MustSlice[float32](want), 1e-5)
}

func TestEinsumTranspose(t *testing.T) {
	a, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, device.CPU())
	got, err := ops.Einsum("ij->ji", a)
	if err != nil {
		t.Fatal(err)
	}
	if got.Dim(0) != 3 || got.Dim(1) != 2 {
		t.Fatalf("shape %v", got.Shape())
	}
	f32Close(t, "ij->ji", tensor.MustSlice[float32](got), []float32{1, 4, 2, 5, 3, 6}, 0)
	// 3-D permutation.
	x := seqTensor(t, []int{2, 3, 4}, 1)
	p, err := ops.Einsum("ijk->kji", x)
	if err != nil {
		t.Fatal(err)
	}
	pv, _ := x.Permute(2, 1, 0)
	pc, _ := pv.Contiguous()
	f32Close(t, "ijk->kji", tensor.MustSlice[float32](p), tensor.MustSlice[float32](pc), 0)
}

func TestEinsumReductions(t *testing.T) {
	a, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6}, []int{2, 3}, device.CPU())
	rows, err := ops.Einsum("ij->i", a)
	if err != nil {
		t.Fatal(err)
	}
	f32Close(t, "ij->i", tensor.MustSlice[float32](rows), []float32{6, 15}, 1e-6)
	cols, _ := ops.Einsum("ij->j", a)
	f32Close(t, "ij->j", tensor.MustSlice[float32](cols), []float32{5, 7, 9}, 1e-6)
	all, _ := ops.Einsum("ij->", a)
	if all.Rank() != 0 {
		t.Errorf("ij-> should be rank-0, got %v", all.Shape())
	}
	f32Close(t, "ij->", tensor.MustSlice[float32](all), []float32{21}, 1e-6)
}

func TestEinsumDotAndOuter(t *testing.T) {
	a, _ := tensor.New([]float32{1, 2, 3}, []int{3}, device.CPU())
	b, _ := tensor.New([]float32{4, 5, 6}, []int{3}, device.CPU())
	dot, err := ops.Einsum("i,i->", a, b)
	if err != nil {
		t.Fatal(err)
	}
	f32Close(t, "i,i->", tensor.MustSlice[float32](dot), []float32{32}, 1e-6)
	c, _ := tensor.New([]float32{10, 20}, []int{2}, device.CPU())
	outer, err := ops.Einsum("i,j->ij", a, c)
	if err != nil {
		t.Fatal(err)
	}
	f32Close(t, "i,j->ij", tensor.MustSlice[float32](outer), []float32{10, 20, 20, 40, 30, 60}, 1e-6)
}

func TestEinsumElementwise(t *testing.T) {
	a, _ := tensor.New([]float32{1, 2, 3, 4}, []int{2, 2}, device.CPU())
	b, _ := tensor.New([]float32{5, 6, 7, 8}, []int{2, 2}, device.CPU())
	got, err := ops.Einsum("ij,ij->ij", a, b)
	if err != nil {
		t.Fatal(err)
	}
	f32Close(t, "ij,ij->ij", tensor.MustSlice[float32](got), []float32{5, 12, 21, 32}, 1e-6)
	// Broadcast-style: scale rows of a by vector.
	v, _ := tensor.New([]float32{10, 100}, []int{2}, device.CPU())
	got, _ = ops.Einsum("ij,i->ij", a, v)
	f32Close(t, "ij,i->ij", tensor.MustSlice[float32](got), []float32{10, 20, 300, 400}, 1e-6)
}

func TestEinsumTraceAndDiagonal(t *testing.T) {
	a, _ := tensor.New([]float32{1, 2, 3, 4, 5, 6, 7, 8, 9}, []int{3, 3}, device.CPU())
	diag, err := ops.Einsum("ii->i", a)
	if err != nil {
		t.Fatal(err)
	}
	f32Close(t, "ii->i", tensor.MustSlice[float32](diag), []float32{1, 5, 9}, 0)
	tr, _ := ops.Einsum("ii->", a)
	f32Close(t, "ii->", tensor.MustSlice[float32](tr), []float32{15}, 0)
}

func TestEinsumThreeOperands(t *testing.T) {
	// ij,jk,kl->il == (A @ B) @ C
	a := seqTensor(t, []int{2, 3}, 0.5)
	b := seqTensor(t, []int{3, 4}, 0.25)
	c := seqTensor(t, []int{4, 2}, 0.5)
	got, err := ops.Einsum("ij,jk,kl->il", a, b, c)
	if err != nil {
		t.Fatal(err)
	}
	ab, _ := ops.MatMul(a, b)
	want, _ := ops.MatMul(ab, c)
	f32Close(t, "ij,jk,kl->il", tensor.MustSlice[float32](got), tensor.MustSlice[float32](want), 1e-5)
}

func TestEinsumAttentionPattern(t *testing.T) {
	// bhqd,bhkd->bhqk — the Q·Kᵀ step of attention, checked against loops.
	B, H, Sq, Sk, D := 2, 2, 3, 4, 5
	q := seqTensor(t, []int{B, H, Sq, D}, 0.1)
	k := seqTensor(t, []int{B, H, Sk, D}, 0.2)
	got, err := ops.Einsum("bhqd,bhkd->bhqk", q, k)
	if err != nil {
		t.Fatal(err)
	}
	qs := tensor.MustSlice[float32](q)
	ks := tensor.MustSlice[float32](k)
	want := make([]float32, B*H*Sq*Sk)
	for b := range B {
		for h := range H {
			for i := range Sq {
				for j := range Sk {
					var s float32
					for d := range D {
						s += qs[((b*H+h)*Sq+i)*D+d] * ks[((b*H+h)*Sk+j)*D+d]
					}
					want[((b*H+h)*Sq+i)*Sk+j] = s
				}
			}
		}
	}
	f32Close(t, "bhqd,bhkd->bhqk", tensor.MustSlice[float32](got), want, 1e-5)
}

func TestEinsumF64AndErrors(t *testing.T) {
	a, _ := tensor.New([]float64{1, 2, 3, 4}, []int{2, 2}, device.CPU())
	b, _ := tensor.New([]float64{5, 6, 7, 8}, []int{2, 2}, device.CPU())
	got, err := ops.Einsum("ij,jk->ik", a, b)
	if err != nil {
		t.Fatal(err)
	}
	if got.DType() != dtype.F64 {
		t.Errorf("dtype %v", got.DType())
	}
	g := tensor.MustSlice[float64](got)
	for i, w := range []float64{19, 22, 43, 50} {
		if math.Abs(g[i]-w) > 1e-12 {
			t.Errorf("[%d] = %v want %v", i, g[i], w)
		}
	}
	if _, err := ops.Einsum("ij,jk", a, b); err == nil {
		t.Error("expected missing-arrow error")
	}
	if _, err := ops.Einsum("ij,jk->ik", a); err == nil {
		t.Error("expected operand-count error")
	}
	if _, err := ops.Einsum("ijk->ij", a); err == nil {
		t.Error("expected rank-mismatch error")
	}
	if _, err := ops.Einsum("ij,jk->ikz", a, b); err == nil {
		t.Error("expected unknown-index error")
	}
	if _, err := ops.Einsum("ij->ii", a); err == nil {
		t.Error("expected repeated-output error")
	}
	c3, _ := tensor.New([]float64{1, 2, 3, 4, 5, 6}, []int{3, 2}, device.CPU())
	if _, err := ops.Einsum("ij,jk->ik", a, c3); err == nil {
		t.Error("expected size-mismatch error")
	}
}
