package models

import "testing"

// TestDDIMTimestepsLeadingMode checks our DDIM timestep selection matches
// HuggingFace Diffusers' default "leading" mode for SD 1.5 (steps_offset 1):
//
//	timesteps = (arange(N) * (1000/N)).round()[::-1] + 1
//
// For N=50 that's [981, 961, 941, ..., 21, 1]. This was off by `step-1`
// in the first cut and visibly hurt image quality; the test locks the fix.
func TestDDIMTimestepsLeadingMode(t *testing.T) {
	cases := []struct {
		numInference int
		want         []int
	}{
		{
			numInference: 50,
			want:         expectedTimesteps(1000, 50),
		},
		{
			numInference: 25,
			want:         expectedTimesteps(1000, 25),
		},
		{
			numInference: 10,
			want:         expectedTimesteps(1000, 10),
		},
	}
	for _, c := range cases {
		s := NewDDIMScheduler(c.numInference)
		if len(s.Timesteps) != c.numInference {
			t.Fatalf("N=%d: got %d timesteps, want %d", c.numInference, len(s.Timesteps), c.numInference)
		}
		for i := range c.want {
			if s.Timesteps[i] != c.want[i] {
				t.Fatalf("N=%d step %d: got %d, want %d (full=%v)",
					c.numInference, i, s.Timesteps[i], c.want[i], s.Timesteps)
			}
		}
		// Bookend: first should be (N-1)*step + 1, last should be 1.
		step := 1000 / c.numInference
		if s.Timesteps[0] != (c.numInference-1)*step+1 {
			t.Fatalf("N=%d first ts=%d, want %d", c.numInference, s.Timesteps[0], (c.numInference-1)*step+1)
		}
		if s.Timesteps[len(s.Timesteps)-1] != 1 {
			t.Fatalf("N=%d last ts=%d, want 1", c.numInference, s.Timesteps[len(s.Timesteps)-1])
		}
	}
}

// expectedTimesteps reproduces HF Diffusers' leading mode formula.
func expectedTimesteps(numTrain, numInference int) []int {
	step := numTrain / numInference
	out := make([]int, numInference)
	for i := 0; i < numInference; i++ {
		out[i] = (numInference-1-i)*step + 1
	}
	return out
}

// TestDDIMAlphasMonotonic verifies the precomputed α_cumprod table is
// monotonically decreasing — a sanity check on the β schedule.
func TestDDIMAlphasMonotonic(t *testing.T) {
	s := NewDDIMScheduler(50)
	for i := 1; i < len(s.AlphasCumprod); i++ {
		if s.AlphasCumprod[i] >= s.AlphasCumprod[i-1] {
			t.Fatalf("alphas_cumprod non-monotonic at %d: %v >= %v",
				i, s.AlphasCumprod[i], s.AlphasCumprod[i-1])
		}
	}
	// SD 1.5's α_cumprod end values matched against HF Diffusers'
	// `DDIMScheduler(beta_schedule="scaled_linear", beta_start=0.00085,
	// beta_end=0.012, num_train_timesteps=1000).alphas_cumprod`. The final
	// cumulative product is ~0.00466 for these constants — not the
	// near-zero one might naively guess.
	if s.AlphasCumprod[0] >= 1.0 || s.AlphasCumprod[0] <= 0.998 {
		t.Fatalf("alphas_cumprod[0]=%v outside expected (0.998, 1.0)", s.AlphasCumprod[0])
	}
	last := s.AlphasCumprod[len(s.AlphasCumprod)-1]
	if last >= 0.005 || last <= 0.004 {
		t.Fatalf("alphas_cumprod[-1]=%v outside expected (0.004, 0.005); got %v", last, last)
	}
}
