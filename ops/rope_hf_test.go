package ops_test

import (
	"math"
	"math/rand/v2"
	"testing"

	"light/device"
	"light/ops"
	"light/tensor"
)

// hfRotateHalfRoPE is a literal transcription of transformers'
// apply_rotary_pos_emb: inv_freq = base^(-arange(0, d, 2)/d), emb = cat(freqs,
// freqs), out = x*cos + rotate_half(x)*sin with rotate_half(x) = cat(-x2, x1).
func hfRotateHalfRoPE(x []float32, S, D int, pos []int64, base float64) []float32 {
	half := D / 2
	out := make([]float32, len(x))
	rows := len(x) / D
	for r := 0; r < rows; r++ {
		p := float64(pos[r%S])
		for j := 0; j < D; j++ {
			i := j % half
			theta := p * math.Pow(base, -float64(2*i)/float64(D))
			c, s := math.Cos(theta), math.Sin(theta)
			var rot float64
			if j < half {
				rot = -float64(x[r*D+j+half])
			} else {
				rot = float64(x[r*D+j-half])
			}
			out[r*D+j] = float32(float64(x[r*D+j])*c + rot*s)
		}
	}
	return out
}

// RoPE must match HuggingFace's rotate-half formula exactly; this is the
// convention every HF Llama-family checkpoint is trained with.
func TestRoPEMatchesHFRotateHalf(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 11))
	const B, H, S, D = 2, 3, 6, 16
	data := make([]float32, B*H*S*D)
	for i := range data {
		data[i] = rng.Float32()*2 - 1
	}
	pos := []int64{3, 4, 5, 6, 7, 8}
	x, _ := tensor.New(data, []int{B, H, S, D}, device.CPU())
	posT, _ := tensor.New(pos, []int{S}, device.CPU())
	want := hfRotateHalfRoPE(data, S, D, pos, 10000)

	got, err := ops.RoPE(x, posT, 10000)
	if err != nil {
		t.Fatal(err)
	}
	g := tensor.MustSlice[float32](got)
	for i := range want {
		if math.Abs(float64(g[i]-want[i])) > 1e-5 {
			t.Fatalf("RoPE[%d] = %v, want %v (HF rotate_half)", i, g[i], want[i])
		}
	}
	inv, _ := tensor.New(baseInvFreq(D, 10000), []int{D / 2}, device.CPU())
	got2, err := ops.RoPEWithInvFreq(x, posT, inv)
	if err != nil {
		t.Fatal(err)
	}
	g2 := tensor.MustSlice[float32](got2)
	for i := range want {
		if math.Abs(float64(g2[i]-want[i])) > 1e-5 {
			t.Fatalf("RoPEWithInvFreq[%d] = %v, want %v (HF rotate_half)", i, g2[i], want[i])
		}
	}
}
