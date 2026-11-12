package diffusion

import (
	"math"
	"testing"
)

// Qwen-Image 2.1's schedule (exponential dynamic shift, max_image_seq_len
// 8192, max_shift 0.9, shift_terminal 0.02) for 8 steps over 4096 tokens,
// as diffusers' FlowMatchEulerDiscreteScheduler.set_timesteps returns it.
func TestFlowMatchShiftTerminal(t *testing.T) {
	term := 0.02
	s := &FlowMatchEuler{NumTrainTimesteps: 1000, Shift: 1, UseDynamicShifting: true, BaseImageSeqLen: 256,
		MaxImageSeqLen: 8192, BaseShift: 0.5, MaxShift: 0.9, ShiftTerminal: &term}
	want := []float64{1.0, 0.916024, 0.8200458, 0.7092946, 0.5800748, 0.4273469, 0.2440539, 0.02, 0.0}
	got := s.Sigmas(8, 4096)
	for i, w := range want {
		if math.Abs(float64(got[i])-w) > 2e-6 {
			t.Fatalf("sigmas %v, want %v", got, want)
		}
	}
}
