// Package diffusion runs diffusers-style pipelines: it loads a pipeline
// folder (model_index.json and one subfolder per component, from the Hub or
// disk), wires the text encoders, denoiser, scheduler and VAE from
// light/models, and generates images. Components live in light/models; this
// package owns the schedulers and the sampling loops.
package diffusion

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
)

// FlowMatchEuler is diffusers' FlowMatchEulerDiscreteScheduler (the
// deterministic Euler path used by Flux and SD3).
type FlowMatchEuler struct {
	NumTrainTimesteps  int     `json:"num_train_timesteps"`
	Shift              float64 `json:"shift"`
	UseDynamicShifting bool    `json:"use_dynamic_shifting"`
	BaseImageSeqLen    int     `json:"base_image_seq_len"`
	MaxImageSeqLen     int     `json:"max_image_seq_len"`
	BaseShift          float64 `json:"base_shift"`
	MaxShift           float64 `json:"max_shift"`
	// ShiftTerminal, when set, stretches the shifted schedule so its last
	// non-zero sigma is this value (stretch_shift_to_terminal).
	ShiftTerminal *float64 `json:"shift_terminal"`
}

// LoadFlowMatchEuler reads a scheduler_config.json.
func LoadFlowMatchEuler(dir string) (*FlowMatchEuler, error) {
	s := &FlowMatchEuler{NumTrainTimesteps: 1000, Shift: 1, BaseImageSeqLen: 256, MaxImageSeqLen: 4096, BaseShift: 0.5, MaxShift: 1.15}
	raw, err := os.ReadFile(filepath.Join(dir, "scheduler_config.json"))
	if err != nil {
		return nil, err
	}
	var probe struct {
		Class         string   `json:"_class_name"`
		Karras        bool     `json:"use_karras_sigmas"`
		Exponential   bool     `json:"use_exponential_sigmas"`
		Beta          bool     `json:"use_beta_sigmas"`
		Invert        bool     `json:"invert_sigmas"`
		Stochastic    bool     `json:"stochastic_sampling"`
		ShiftTerminal *float64 `json:"shift_terminal"`
		TimeShiftType string   `json:"time_shift_type"`
	}
	if err := json.Unmarshal(raw, &probe); err != nil {
		return nil, fmt.Errorf("scheduler config: %w", err)
	}
	if probe.Class != "" && probe.Class != "FlowMatchEulerDiscreteScheduler" {
		return nil, fmt.Errorf("scheduler %q is not FlowMatchEulerDiscreteScheduler", probe.Class)
	}
	if probe.Karras || probe.Exponential || probe.Beta || probe.Invert || probe.Stochastic ||
		(probe.TimeShiftType != "" && probe.TimeShiftType != "exponential") {
		return nil, fmt.Errorf("FlowMatchEuler: only the plain (optionally shifted) schedule is supported")
	}
	if err := json.Unmarshal(raw, s); err != nil {
		return nil, err
	}
	return s, nil
}

// stretch is stretch_shift_to_terminal on the first n sigmas (F32).
func (s *FlowMatchEuler) stretch(sig []float32, n int) {
	if s.ShiftTerminal == nil || *s.ShiftTerminal == 0 || n == 0 {
		return
	}
	scale := (1 - sig[n-1]) / float32(1-*s.ShiftTerminal)
	for i := range n {
		sig[i] = 1 - (1-sig[i])/scale
	}
}

// SigmasMu is Sigmas with an explicit time-shift mu (FLUX.2's empirical
// mu from the sequence length and step count) under dynamic shifting.
func (s *FlowMatchEuler) SigmasMu(n int, mu float64) []float32 {
	out := make([]float32, n+1)
	for i := range n {
		var lin float64 = 1
		if n > 1 {
			lin = 1 + (1/float64(n)-1)*float64(i)/float64(n-1)
		}
		sg := float32(lin)
		if s.UseDynamicShifting {
			e := float32(math.Exp(mu))
			sg = e / (e + (1/sg - 1))
		} else {
			sh := float32(s.Shift)
			sg = sh * sg / (1 + (sh-1)*sg)
		}
		out[i] = sg
	}
	s.stretch(out, n)
	return out
}

// Sigmas is the schedule for n steps over imageSeqLen latent tokens, as
// FluxPipeline sets it: linspace(1, 1/n, n), shifted (exponential time
// shift with mu from the sequence length, or the static shift), in F32,
// followed by the terminal 0.
func (s *FlowMatchEuler) Sigmas(n, imageSeqLen int) []float32 {
	out := make([]float32, n+1)
	var mu float64
	if s.UseDynamicShifting {
		m := (s.MaxShift - s.BaseShift) / float64(s.MaxImageSeqLen-s.BaseImageSeqLen)
		mu = float64(imageSeqLen)*m + (s.BaseShift - m*float64(s.BaseImageSeqLen))
	}
	for i := range n {
		var lin float64 = 1
		if n > 1 {
			lin = 1 + (1/float64(n)-1)*float64(i)/float64(n-1)
		}
		sg := float32(lin) // np.array(sigmas).astype(np.float32)
		if s.UseDynamicShifting {
			e := float32(math.Exp(mu))
			sg = e / (e + (1/sg - 1))
		} else {
			sh := float32(s.Shift)
			sg = sh * sg / (1 + (sh-1)*sg)
		}
		out[i] = sg
	}
	s.stretch(out, n)
	return out
}
