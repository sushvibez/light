package models

import (
	"light/ops"
	"light/tensor"
)

// IdolSampleOptions control IdolSample (idol's sample_flow).
type IdolSampleOptions struct {
	Steps int     // default 50
	CFG   float64 // guidance strength w, default 3 (0 disables)
	Shift float64 // time shift (rescale_t), default 3
	// Step, when set, is called after each step (progress).
	Step func(i, n int)
}

// IdolSample integrates the denoiser from noise (t = 1) to data (t = 0)
// with Euler steps over the shifted grid t' = s·t / (1 + (s − 1)·t) and
// classifier-free guidance against zeroed image tokens:
// v = (1 + w)·v(cond) − w·v(0). Returns the latent and the camera.
func IdolSample(m *IdolDenoiser, latent, camera, feature1, feature2 *tensor.Tensor, o IdolSampleOptions) (*tensor.Tensor, *tensor.Tensor, error) {
	if o.Steps == 0 {
		o.Steps = 50
	}
	if o.Shift == 0 {
		o.Shift = 3
	}
	var zero1, zero2 *tensor.Tensor
	var err error
	if o.CFG > 0 {
		if zero1, err = tensor.Zeros(feature1.DType(), feature1.Shape().Dims(), feature1.Device()); err != nil {
			return nil, nil, err
		}
		if zero2, err = tensor.Zeros(feature2.DType(), feature2.Shape().Dims(), feature2.Device()); err != nil {
			return nil, nil, err
		}
	}
	grid := make([]float64, o.Steps+1)
	for i := range grid {
		t := 1 - float64(i)/float64(o.Steps)
		grid[i] = o.Shift * t / (1 + (o.Shift-1)*t)
	}
	B := latent.Dim(0)
	x, c := latent, camera
	for i := range o.Steps {
		t, tPrev := grid[i], grid[i+1]
		ts := make([]float32, B)
		for b := range ts {
			ts[b] = float32(1000 * t)
		}
		vx, vc, err := m.Forward(IdolInputs{Latent: x, Camera: c, T: ts, Feature1: feature1, Feature2: feature2})
		if err != nil {
			return nil, nil, err
		}
		if o.CFG > 0 {
			ux, uc, err := m.Forward(IdolInputs{Latent: x, Camera: c, T: ts, Feature1: zero1, Feature2: zero2})
			if err != nil {
				return nil, nil, err
			}
			if vx, err = idolGuide(vx, ux, o.CFG); err != nil {
				return nil, nil, err
			}
			if vc, err = idolGuide(vc, uc, o.CFG); err != nil {
				return nil, nil, err
			}
		}
		dt := t - tPrev
		if x, err = idolEuler(x, vx, dt); err != nil {
			return nil, nil, err
		}
		if c, err = idolEuler(c, vc, dt); err != nil {
			return nil, nil, err
		}
		if o.Step != nil {
			o.Step(i, o.Steps)
		}
	}
	return x, c, nil
}

// idolGuide is (1 + w)·v − w·u.
func idolGuide(v, u *tensor.Tensor, w float64) (*tensor.Tensor, error) {
	a, err := ops.Affine(v, 1+w, 0)
	if err != nil {
		return nil, err
	}
	b, err := ops.Affine(u, w, 0)
	if err != nil {
		return nil, err
	}
	return ops.Sub(a, b)
}

// idolEuler is x − v·dt.
func idolEuler(x, v *tensor.Tensor, dt float64) (*tensor.Tensor, error) {
	s, err := ops.Affine(v, dt, 0)
	if err != nil {
		return nil, err
	}
	return ops.Sub(x, s)
}
