package models

import (
	"fmt"
	"os"

	"light/device"
	"light/dtype"
	"light/hub"
	"light/tokenizer"
)

// LoadSDXLFromHF loads SDXL base from the Hub (fp16 variant files): both
// text encoders at F32, the UNet at wdt (BF16 recommended: 5.2 GB), the VAE
// decoder at F32. ~9 GB on the card at BF16 with the 1024² activations.
func LoadSDXLFromHF(repo string, wdt dtype.DType, dev device.Device) (*StableDiffusionXL, error) {
	if repo == "" {
		repo = SDXLRepo
	}
	tok1, err := loadCLIPTokenizerDir(repo, "tokenizer")
	if err != nil {
		return nil, err
	}
	tok2, err := loadCLIPTokenizerDir(repo, "tokenizer_2")
	if err != nil {
		return nil, err
	}
	te1, err := loadCLIPTextVariant(repo, "text_encoder", "fp16", CLIPTextSDXL1(), dev)
	if err != nil {
		return nil, fmt.Errorf("text encoder 1: %w", err)
	}
	te2, err := loadCLIPTextVariant(repo, "text_encoder_2", "fp16", CLIPTextSDXL2(), dev)
	if err != nil {
		return nil, fmt.Errorf("text encoder 2: %w", err)
	}
	weights, err := loadComponentSafetensorsVariant(repo, "unet", "fp16", dev)
	if err != nil {
		return nil, fmt.Errorf("unet: %w", err)
	}
	cfg := SDUNetSDXL()
	cfg.WeightDType = wdt
	unet, err := loadSDUNetWeights(cfg, weights, dev)
	if err != nil {
		return nil, fmt.Errorf("unet: %w", err)
	}
	weights = nil
	vae, _, err := loadSDVAEVariant(repo, "fp16", dtype.F32, dev)
	if err != nil {
		return nil, fmt.Errorf("vae: %w", err)
	}
	return &StableDiffusionXL{
		TextEncoder1: te1, TextEncoder2: te2, Tokenizer1: tok1, Tokenizer2: tok2,
		UNet: unet, VAE: vae, Scheduler: NewDDIMScheduler(30),
	}, nil
}

func loadCLIPTokenizerDir(repo, subdir string) (*tokenizer.CLIPTokenizer, error) {
	vocabPath, err := hub.Download(repo, "main", subdir+"/vocab.json")
	if err != nil {
		return nil, fmt.Errorf("%s/vocab.json: %w", subdir, err)
	}
	mergesPath, err := hub.Download(repo, "main", subdir+"/merges.txt")
	if err != nil {
		return nil, fmt.Errorf("%s/merges.txt: %w", subdir, err)
	}
	vocabF, err := os.Open(vocabPath)
	if err != nil {
		return nil, err
	}
	defer vocabF.Close()
	mergesF, err := os.Open(mergesPath)
	if err != nil {
		return nil, err
	}
	defer mergesF.Close()
	return tokenizer.LoadCLIPTokenizer(vocabF, mergesF)
}

// loadCLIPTextVariant loads a CLIPTextModel(WithProjection) checkpoint
// (HF layout) into a CLIPText of cfg.
func loadCLIPTextVariant(repo, subdir, variant string, cfg CLIPTextConfig, dev device.Device) (*CLIPText, error) {
	weights, err := loadComponentSafetensorsVariant(repo, subdir, variant, dev)
	if err != nil {
		return nil, err
	}
	te, err := NewCLIPText(cfg, true, dev)
	if err != nil {
		return nil, err
	}
	if err := loadCLIPTextWeights(te, weights); err != nil {
		return nil, err
	}
	return te, nil
}
