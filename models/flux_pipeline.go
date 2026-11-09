package models

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"light/device"
	"light/dtype"
)

// FluxVAE is a Flux / SD3 AutoencoderKL decoder with its latent
// normalisation (latents are divided by ScalingFactor and shifted by
// ShiftFactor before decoding).
type FluxVAE struct {
	Decoder       *VAEDecoder
	ScalingFactor float32
	ShiftFactor   float32
}

type autoencoderKLConfig struct {
	LatentChannels   int     `json:"latent_channels"`
	BlockOut         []int   `json:"block_out_channels"`
	LayersPerBlock   int     `json:"layers_per_block"`
	UsePostQuantConv *bool   `json:"use_post_quant_conv"`
	ScalingFactor    float32 `json:"scaling_factor"`
	ShiftFactor      float32 `json:"shift_factor"`
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
	return &FluxVAE{Decoder: d, ScalingFactor: c.ScalingFactor, ShiftFactor: c.ShiftFactor}, nil
}
