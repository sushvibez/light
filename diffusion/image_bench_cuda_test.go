//go:build !nocuda && cgo

package diffusion

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"light/device"
	"light/tensor"
)

// TestImageBenchCUDA times full generations (loads included, sequential
// loading, 16-bit storage) for docs/image-bench.md. LIGHT_IMAGE_BENCH
// lists runs separated by ";": kind,repo,transformer,size,steps,guidance
// (transformer "" = the repo's; "all" = the built-in list below).
//
//	LIGHT_HEAVY_TESTS=1 LIGHT_IMAGE_BENCH=all go test ./diffusion -run TestImageBenchCUDA -v -timeout 3h
func TestImageBenchCUDA(t *testing.T) {
	if os.Getenv("LIGHT_HEAVY_TESTS") == "" || os.Getenv("LIGHT_IMAGE_BENCH") == "" {
		t.Skip("set LIGHT_HEAVY_TESTS=1 LIGHT_IMAGE_BENCH=all|<runs>")
	}
	dev, err := device.CUDA(0)
	if err != nil {
		t.Skip(err)
	}
	home, _ := os.UserHomeDir()
	lib := filepath.Join(home, ".cache", "light", "models", "checkpoints")
	if d := os.Getenv("LIGHT_MODELS_DIR"); d != "" {
		lib = filepath.Join(d, "checkpoints")
	}
	spec := os.Getenv("LIGHT_IMAGE_BENCH")
	if spec == "all" {
		spec = strings.Join([]string{
			"zimage,Tongyi-MAI/Z-Image-Turbo,,1024,8,0",
			"flux2,black-forest-labs/FLUX.2-klein-4B,,1024,4,0",
			"flux2,black-forest-labs/FLUX.2-klein-4B," + lib + "/flux2-klein-4b-fp8,1024,4,0",
			"flux,black-forest-labs/FLUX.1-schnell," + lib + "/flux1-schnell-fp8.safetensors,1024,4,0",
			"flux,black-forest-labs/FLUX.1-schnell," + lib + "/flux1-schnell-Q4_K_S.gguf,1024,4,0",
			"flux,black-forest-labs/FLUX.1-dev," + lib + "/flux1-dev-fp8.safetensors,1024,28,3.5",
			"flux,black-forest-labs/FLUX.1-dev," + lib + "/flux1-dev-Q4_K_S.gguf,1024,28,3.5",
			"flux,black-forest-labs/FLUX.1-schnell,,1024,4,0",
		}, ";")
	}
	prompt := "a watercolour painting of a red lighthouse on a rocky shore at dawn, soft light"
	var rows []string
	for _, r := range strings.Split(spec, ";") {
		f := strings.Split(strings.TrimSpace(r), ",")
		if len(f) != 6 {
			t.Fatalf("run %q: want kind,repo,transformer,size,steps,guidance", r)
		}
		var size, steps int
		var guidance float32
		fmt.Sscan(f[3], &size)
		fmt.Sscan(f[4], &steps)
		fmt.Sscan(f[5], &guidance)
		kind, repo, tf := f[0], f[1], f[2]
		name := repo[strings.LastIndex(repo, "/")+1:]
		variant := "BF16"
		if tf != "" {
			switch {
			case strings.HasSuffix(tf, ".gguf"):
				variant = "Q4_K_S GGUF"
			case strings.Contains(tf, "fp8"):
				variant = "FP8"
			}
		}
		t0 := time.Now()
		var stepTimes []time.Duration
		last := t0
		var gen func() error
		switch kind {
		case "flux":
			lo := FluxLoadOptions{Storage16: true, Sequential: true}
			if tf != "" {
				lo.Components = map[string]string{"transformer": tf}
				lo.KeepFP8 = true
			} else {
				lo.StreamBlocks = true
				variant = "BF16 streamed"
			}
			gen = func() error {
				p, err := LoadFluxOpts(repo, dev, lo)
				if err != nil {
					return err
				}
				_, err = p.Generate(FluxOptions{Prompt: prompt, Width: size, Height: size, Steps: steps, Guidance: guidance, Seed: 7,
					OnStep: func(int, *tensor.Tensor, *tensor.Tensor) {
						now := time.Now()
						stepTimes = append(stepTimes, now.Sub(last))
						last = now
					}})
				return err
			}
		case "flux2":
			lo := Flux2LoadOptions{Storage16: true, Sequential: true}
			if tf != "" {
				lo.Components = map[string]string{"transformer": tf}
				lo.KeepFP8 = true
			}
			gen = func() error {
				p, err := LoadFlux2(repo, dev, lo)
				if err != nil {
					return err
				}
				_, err = p.Generate(Flux2Options{Prompt: prompt, Width: size, Height: size, Steps: steps, Guidance: guidance, Seed: 7})
				return err
			}
		case "zimage":
			gen = func() error {
				p, err := LoadZImage(repo, dev, ZImageLoadOptions{Storage16: true, Sequential: true})
				if err != nil {
					return err
				}
				_, err = p.Generate(ZImageOptions{Prompt: prompt, Width: size, Height: size, Steps: steps, Guidance: guidance, Seed: 7})
				return err
			}
		default:
			t.Fatalf("kind %q", kind)
		}
		if err := gen(); err != nil {
			t.Errorf("%s %s: %v", name, variant, err)
			rows = append(rows, fmt.Sprintf("| %s | %s | %d² | %d | failed: %v |", name, variant, size, steps, err))
			continue
		}
		total := time.Since(t0)
		per := ""
		if len(stepTimes) > 1 {
			var sum time.Duration
			for _, d := range stepTimes[1:] {
				sum += d
			}
			per = fmt.Sprintf("%.1f s", (sum / time.Duration(len(stepTimes)-1)).Seconds())
		}
		row := fmt.Sprintf("| %s | %s | %d² | %d | %s | %s |", name, variant, size, steps, per, total.Round(time.Second))
		t.Log(row)
		rows = append(rows, row)
		releaseMemory()
	}
	t.Logf("\n| Model | Transformer | Size | Steps | Per step | Full generation |\n|---|---|---|---|---|---|\n%s", strings.Join(rows, "\n"))
}
