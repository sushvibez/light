package models

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"

	"light/device"
	"light/dtype"
	"light/ops"
	"light/tensor"
)

// FluxVAE is a Flux / SD3 AutoencoderKL decoder with its latent
// normalisation (latents are divided by ScalingFactor and shifted by
// ShiftFactor before decoding).
type FluxVAE struct {
	Decoder       *VAEDecoder
	Encoder       *VAEEncoder // nil when the checkpoint has no encoder weights
	ScalingFactor float32
	ShiftFactor   float32
	// FLUX.2 (AutoencoderKLFlux2): the BatchNorm statistics of the 2 x 2
	// patchified latents (running mean, sqrt(running var + eps)); nil
	// otherwise.
	BNMean, BNStd []float32
}

type autoencoderKLConfig struct {
	LatentChannels   int     `json:"latent_channels"`
	BlockOut         []int   `json:"block_out_channels"`
	LayersPerBlock   int     `json:"layers_per_block"`
	UsePostQuantConv *bool   `json:"use_post_quant_conv"`
	UseQuantConv     *bool   `json:"use_quant_conv"`
	ScalingFactor    float32 `json:"scaling_factor"`
	ShiftFactor      float32 `json:"shift_factor"`
	BatchNormEps     float64 `json:"batch_norm_eps"`
}

// LoadFluxVAE loads the decoder of an AutoencoderKL from a directory with
// config.json and its safetensors (a pipeline's `vae/` folder), at F32.
func LoadFluxVAE(dir string, dev device.Device) (*FluxVAE, error) {
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	var c autoencoderKLConfig
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, fmt.Errorf("vae config: %w", err)
	}
	cfg := VAEDecoderConfig{LatentChannels: c.LatentChannels, BlockOut: c.BlockOut, LayersPerBlock: c.LayersPerBlock, PostQuantConv: true}
	if c.UsePostQuantConv != nil {
		cfg.PostQuantConv = *c.UsePostQuantConv
	}
	w, err := loadCheckpointTensors(dir)
	if err != nil {
		return nil, err
	}
	d, err := loadVAEDecoder(w, cfg, dtype.F32, dev)
	if err != nil {
		return nil, fmt.Errorf("LoadFluxVAE: %w", err)
	}
	v := &FluxVAE{Decoder: d, ScalingFactor: c.ScalingFactor, ShiftFactor: c.ShiftFactor}
	if _, ok := w["encoder.conv_in.weight"]; ok {
		qc := c.UseQuantConv == nil || *c.UseQuantConv
		if v.Encoder, err = loadVAEEncoder(w, cfg, qc, dtype.F32, dev); err != nil {
			return nil, fmt.Errorf("LoadFluxVAE: %w", err)
		}
	}
	if mean, ok := w["bn.running_mean"]; ok {
		vr, ok := w["bn.running_var"]
		if !ok {
			return nil, fmt.Errorf("LoadFluxVAE: bn.running_mean without bn.running_var")
		}
		f32 := func(t *tensor.Tensor) ([]float32, error) {
			c, err := ops.Cast(t, dtype.F32)
			if err != nil {
				return nil, err
			}
			if c, err = c.To(device.CPU()); err != nil {
				return nil, err
			}
			return append([]float32(nil), tensor.MustSlice[float32](c)...), nil
		}
		if v.BNMean, err = f32(mean); err != nil {
			return nil, err
		}
		vs, err := f32(vr)
		if err != nil {
			return nil, err
		}
		v.BNStd = make([]float32, len(vs))
		for i, x := range vs {
			v.BNStd[i] = float32(math.Sqrt(float64(x + float32(c.BatchNormEps))))
		}
	}
	return v, nil
}
