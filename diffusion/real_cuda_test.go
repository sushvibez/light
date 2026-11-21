//go:build !nocuda && cgo

package diffusion

import (
	"encoding/json"
	"errors"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"light/device"
	"light/models"
	"light/safetensors"
	"light/tensor"
)

// Real-weight checks against diffusers on the GPU (BF16 both sides). The
// reference comes from scripts/verify/ref_real_pipe.py; point the env var
// at its JSON:
//
//	LIGHT_HEAVY_TESTS=1 LIGHT_ZIMAGE_REAL_REF=/path/zimage.json go test ./diffusion -run TestZImageTurboRealCUDA -v
//	LIGHT_HEAVY_TESTS=1 LIGHT_KLEIN_REAL_REF=/path/klein.json go test ./diffusion -run TestFlux2KleinRealCUDA -v
//	LIGHT_HEAVY_TESTS=1 LIGHT_FLUX1_REAL_REF=/path/schnell.json go test ./diffusion -run TestFlux1RealCUDA -v
//
// Four BF16 denoising steps on both sides accumulate rounding, so the
// images are compared by mean absolute pixel difference (tolerance 0.03
// of the [0, 1] range) and the max is only reported.
type realRef struct {
	Repo         string    `json:"repo"`
	Prompt       string    `json:"prompt"`
	Height       int       `json:"height"`
	Width        int       `json:"width"`
	Steps        int       `json:"steps"`
	Guidance     float32   `json:"guidance"`
	Latents      []float32 `json:"latents"`
	LatentsShape []int     `json:"latents_shape"`
	Image        []float32 `json:"image"`
	ImageShape   []int     `json:"image_shape"`
	// Intermediates (FLUX.1 references): text conditioning, the packed
	// latents after step 0 and the sigma schedule.
	PromptEmbeds []float32 `json:"prompt_embeds"`
	Pooled       []float32 `json:"pooled"`
	LatentsStep0 []float32 `json:"latents_step0"`
	Sigmas       []float32 `json:"sigmas"`
}

// stats logs mean / max |got - want| for an intermediate.
func stats(t *testing.T, name string, got *tensor.Tensor, want []float32) {
	t.Helper()
	c, err := got.To(device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	g := tensor.MustSlice[float32](c)
	if len(g) != len(want) {
		t.Errorf("%s: %d values, reference %d", name, len(g), len(want))
		return
	}
	var sum, mx, ref, refMean float64
	for i := range g {
		d := math.Abs(float64(g[i] - want[i]))
		sum += d
		mx = math.Max(mx, d)
		ref = math.Max(ref, math.Abs(float64(want[i])))
		refMean += math.Abs(float64(want[i]))
	}
	n := float64(len(g))
	t.Logf("%s: mean |Δ| %.4g (%.2f%% of mean |x| %.3g), max %.4g (max |x| %.3g)", name, sum/n, 100*sum/refMean, refMean/n, mx, ref)
}

func loadRealRef(t *testing.T, env string) (realRef, device.Device) {
	t.Helper()
	if os.Getenv("LIGHT_HEAVY_TESTS") == "" {
		t.Skip("set LIGHT_HEAVY_TESTS=1")
	}
	p := os.Getenv(env)
	if p == "" {
		t.Skipf("set %s to the reference JSON (scripts/verify/ref_real_pipe.py)", env)
	}
	if !device.CUDAAvailable() {
		t.Skip("no CUDA")
	}
	dev, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(p)
	if err != nil {
		t.Fatal(err)
	}
	var r realRef
	if err := json.Unmarshal(raw, &r); err != nil {
		t.Fatal(err)
	}
	return r, dev
}

func compareReal(t *testing.T, got *tensor.Tensor, ref realRef) {
	compareRealTol(t, got, ref, 0.03)
}

// compareRealTol is compareReal with its own bar on the mean pixel
// difference (a 4-bit transformer sits above the 16-bit one).
func compareRealTol(t *testing.T, got *tensor.Tensor, ref realRef, tol float64) {
	t.Helper()
	c, err := got.To(device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	g := tensor.MustSlice[float32](c)
	if len(g) != len(ref.Image) {
		t.Fatalf("image %v, want %v", got.Shape(), ref.ImageShape)
	}
	var sum, mx float64
	for i := range g {
		d := math.Abs(float64(g[i] - ref.Image[i]))
		sum += d
		mx = math.Max(mx, d)
	}
	mean := sum / float64(len(g))
	t.Logf("%s: %dx%d, %d steps: mean |Δ| = %.4f, max %.3f (of 1.0)", ref.Repo, ref.Width, ref.Height, ref.Steps, mean, mx)
	if dir := os.Getenv("LIGHT_REAL_SAVE_DIR"); dir != "" { // both images, for a look
		name := strings.ReplaceAll(ref.Repo, "/", "--")
		if err := (Image{Pixels: got}).SavePNG(filepath.Join(dir, name+"-light.png")); err != nil {
			t.Error(err)
		}
		r, _ := tensor.New(ref.Image, append([]int{1}, ref.ImageShape...), device.CPU())
		if err := (Image{Pixels: r}).SavePNG(filepath.Join(dir, name+"-diffusers.png")); err != nil {
			t.Error(err)
		}
	}
	if mean > tol {
		t.Errorf("mean pixel difference %.4f > %.2f", mean, tol)
	}
}

func TestZImageTurboRealCUDA(t *testing.T) {
	ref, dev := loadRealRef(t, "LIGHT_ZIMAGE_REAL_REF")
	p, err := LoadZImage(ref.Repo, dev, ZImageLoadOptions{Storage16: true, Sequential: true})
	if err != nil {
		t.Fatal(err)
	}
	lat, err := tensor.New(ref.Latents, ref.LatentsShape[1:], dev) // (C, h, w): the batch axis dropped
	if err != nil {
		t.Fatal(err)
	}
	res, err := p.Generate(ZImageOptions{Prompt: ref.Prompt, Width: ref.Width, Height: ref.Height, Steps: ref.Steps, Guidance: ref.Guidance, Latents: lat})
	if err != nil {
		t.Fatal(err)
	}
	compareReal(t, res.Image, ref)
}

func TestFlux2KleinRealCUDA(t *testing.T) {
	ref, dev := loadRealRef(t, "LIGHT_KLEIN_REAL_REF")
	lo := Flux2LoadOptions{Storage16: true, Sequential: true}
	tol := 0.03
	if tf := os.Getenv("LIGHT_FLUX2_TRANSFORMER"); tf != "" { // an FP8 / quantized transformer folder in place of the repo's
		lo.Components = map[string]string{"transformer": tf}
		lo.KeepFP8 = true
		tol = 0.05
		t.Logf("transformer: %s", tf)
	}
	if os.Getenv("LIGHT_FLUX2_ENCODER_CPU") != "" { // the Mistral-3 24B encoder of FLUX.2-dev beside a 40 GB card
		lo.EncoderDevice = device.CPU()
	}
	if os.Getenv("LIGHT_FLUX2_STREAM") != "" { // transformer blocks streamed (a 24 GB card)
		lo.StreamBlocks = true
	}
	if os.Getenv("LIGHT_FLUX_MIXED16") != "" { // BF16 tensor-core compute
		lo.Mixed16 = true
		tol += 0.01
		t.Log("Mixed16: BF16 compute")
	}
	if ref.Steps > 4 {
		tol += 0.015
	}
	t0 := time.Now()
	p, err := LoadFlux2(ref.Repo, dev, lo)
	if err != nil {
		t.Fatal(err)
	}
	lat, err := tensor.New(ref.Latents, ref.LatentsShape, dev)
	if err != nil {
		t.Fatal(err)
	}
	res, err := p.Generate(Flux2Options{Prompt: ref.Prompt, Width: ref.Width, Height: ref.Height, Steps: ref.Steps, Guidance: ref.Guidance, Latents: lat})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("load + generate: %s", time.Since(t0).Round(100*time.Millisecond))
	compareRealTol(t, res.Image, ref, tol)
}

// FLUX.1 (schnell or dev): the 12B transformer streams its blocks from the
// memory-mapped checkpoint, so the run fits a 16 GB card.
func TestFlux1RealCUDA(t *testing.T) {
	ref, dev := loadRealRef(t, "LIGHT_FLUX1_REAL_REF")
	slots, _ := strconv.Atoi(os.Getenv("LIGHT_FLUX_SLOTS"))
	lo := FluxLoadOptions{Storage16: true, Sequential: true, StreamBlocks: true, StreamSlots: slots}
	// LIGHT_FLUX1_TRANSFORMER: a single-file transformer (BFL / diffusers
	// names, BF16, FP8 or GGUF) in place of the repo's; FP8 and GGUF ones
	// fit the card resident, so they are not streamed unless
	// LIGHT_FLUX1_STREAM=1.
	if tf := os.Getenv("LIGHT_FLUX1_TRANSFORMER"); tf != "" {
		lo.Components = map[string]string{"transformer": tf}
		lo.KeepFP8 = true
		lo.StreamBlocks = os.Getenv("LIGHT_FLUX1_STREAM") != ""
		t.Logf("transformer: %s (streamed: %v)", tf, lo.StreamBlocks)
	}
	mixed := os.Getenv("LIGHT_FLUX_MIXED16") != ""
	lo.Mixed16 = mixed
	p, err := LoadFluxOpts(ref.Repo, dev, lo)
	if err != nil {
		t.Fatal(err)
	}
	noise, err := tensor.New(ref.Latents, ref.LatentsShape, dev) // (1, C, h, w) before packing
	if err != nil {
		t.Fatal(err)
	}
	seq := 512
	if strings.HasSuffix(ref.Repo, "schnell") {
		seq = 256
	}
	var step0, vel0 *tensor.Tensor
	t0 := time.Now()
	var stepTimes []time.Duration
	last := t0
	res, err := p.Generate(FluxOptions{Prompt: ref.Prompt, Width: ref.Width, Height: ref.Height, Steps: ref.Steps, Guidance: ref.Guidance, MaxSeqLen: seq, Noise: noise,
		OnStep: func(i int, x, v *tensor.Tensor) {
			if i == 0 {
				step0, vel0 = x, v
			}
			now := time.Now()
			stepTimes = append(stepTimes, now.Sub(last).Round(10*time.Millisecond))
			last = now
		}})
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("generate: %s in all; step times (the first includes loading) %v", time.Since(t0).Round(100*time.Millisecond), stepTimes)
	if ref.LatentsStep0 != nil && vel0 != nil && len(ref.Sigmas) > 1 {
		// The reference's velocity: (x1 - x0) / (σ1 - σ0) with x0 the packed noise.
		C, h, w := ref.LatentsShape[1], ref.LatentsShape[2], ref.LatentsShape[3]
		n, _ := tensor.New(ref.Latents, []int{1, C, h / 2, 2, w / 2, 2}, device.CPU())
		n, _ = n.Permute(0, 2, 4, 1, 3, 5)
		n, _ = n.Contiguous()
		x0 := tensor.MustSlice[float32](n)
		vref := make([]float32, len(x0))
		d := ref.Sigmas[1] - ref.Sigmas[0]
		var sv, sl float64
		for i := range x0 {
			vref[i] = (ref.LatentsStep0[i] - x0[i]) / d
			sv += math.Abs(float64(vref[i]))
		}
		vc, _ := vel0.To(device.CPU())
		for _, v := range tensor.MustSlice[float32](vc) {
			sl += math.Abs(float64(v))
		}
		t.Logf("velocity at step 0: mean |v| light %.4g, reference %.4g", sl/float64(len(x0)), sv/float64(len(x0)))
		stats(t, "velocity at step 0", vel0, vref)
	}
	if ref.Sigmas != nil {
		t.Logf("sigmas: light %v, reference %v", res.Sigmas, ref.Sigmas)
	}
	if ref.PromptEmbeds != nil {
		stats(t, "T5 states", res.PromptEmbeds, ref.PromptEmbeds)
		stats(t, "CLIP pooled", res.Pooled, ref.Pooled)
	}
	if ref.LatentsStep0 != nil && step0 != nil {
		stats(t, "latents after step 0", step0, ref.LatentsStep0)
	}
	tol := 0.03
	if ref.Steps > 4 {
		tol = 0.045 // BF16 rounding on both sides compounds per step (dev, 8 steps: 0.036)
	}
	if mixed {
		tol += 0.01 // BF16 compute, as the reference itself
	}
	if strings.HasSuffix(strings.ToLower(os.Getenv("LIGHT_FLUX1_TRANSFORMER")), ".gguf") {
		tol = 0.05 // 4-bit weights: 0.033 for Q4_K_S schnell, the same scene
		if ref.Steps > 4 {
			tol = 0.08 // dev, 8 steps: 0.066, the scene with the rocks redrawn
		}
	}
	compareRealTol(t, res.Image, ref, tol)
}

// The text encoders alone against the reference's prompt_embeds / pooled,
// per position (padding included: FLUX attends to every text token).
func TestFlux1TextEncodersRealCUDA(t *testing.T) {
	ref, dev := loadRealRef(t, "LIGHT_FLUX1_REAL_REF")
	if ref.PromptEmbeds == nil {
		t.Skip("reference without prompt_embeds")
	}
	p, err := LoadFluxOpts(ref.Repo, dev, FluxLoadOptions{Storage16: true, Sequential: true})
	if err != nil {
		t.Fatal(err)
	}
	if err := p.loadEncoders(); err != nil {
		t.Fatal(err)
	}
	seq := 512
	if strings.HasSuffix(ref.Repo, "schnell") {
		seq = 256
	}
	tids, err := p.t5IDs(ref.Prompt, seq)
	if err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, id := range tids {
		if id != 0 {
			n++
		}
	}
	tt, _ := tensor.New(tids, []int{1, len(tids)}, dev)
	emb, err := p.T5.Forward(tt)
	if err != nil {
		t.Fatal(err)
	}
	c, _ := emb.To(device.CPU())
	g := tensor.MustSlice[float32](c)
	D := len(g) / seq
	t.Logf("T5: %d real tokens of %d, ids %v", n, seq, tids[:n+1])
	for _, pos := range []int{0, 1, n / 2, n - 1, n, n + 1, seq / 2, seq - 1} {
		var sum, mx, ref2 float64
		for j := range D {
			d := math.Abs(float64(g[pos*D+j] - ref.PromptEmbeds[pos*D+j]))
			sum += d
			mx = math.Max(mx, d)
			ref2 += float64(ref.PromptEmbeds[pos*D+j] * ref.PromptEmbeds[pos*D+j])
		}
		t.Logf("T5 pos %3d: mean |Δ| %.4g, max %.4g, reference rms %.3g", pos, sum/float64(D), mx, math.Sqrt(ref2/float64(D)))
	}
}

// One forward pass of the real transformer at step 0, block by block,
// against scripts/verify/ref_flux_blocks.py (LIGHT_FLUX1_BLOCKS_REF).
func TestFlux1BlocksRealCUDA(t *testing.T) {
	ref, dev := loadRealRef(t, "LIGHT_FLUX1_REAL_REF")
	bp := os.Getenv("LIGHT_FLUX1_BLOCKS_REF")
	if bp == "" || ref.PromptEmbeds == nil {
		t.Skip("set LIGHT_FLUX1_BLOCKS_REF (ref_flux_blocks.py) and a reference with prompt_embeds")
	}
	if os.Getenv("LIGHT_FLUX1_CPU") != "" { // the same on the CPU (slow; stops after block 0)
		dev = device.CPU()
	}
	stopAfter0 := os.Getenv("LIGHT_FLUX1_BLOCK0_ONLY") != ""
	errStop := errors.New("stop after block 0")
	f, err := os.Open(bp)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	tr, _, err := safetensors.Load(f, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	p, err := LoadFluxOpts(ref.Repo, dev, FluxLoadOptions{Storage16: true, Sequential: true, StreamBlocks: true})
	if err != nil {
		t.Fatal(err)
	}
	if err := p.loadTransformer(); err != nil {
		t.Fatal(err)
	}
	m := p.Transformer
	var x *tensor.Tensor
	if xt, ok := tr["x"]; ok {
		x, _ = xt.To(dev)
	} else { // pack the reference noise
		C, h, w := ref.LatentsShape[1], ref.LatentsShape[2], ref.LatentsShape[3]
		n, _ := tensor.New(ref.Latents, []int{1, C, h / 2, 2, w / 2, 2}, dev)
		n, _ = n.Permute(0, 2, 4, 1, 3, 5)
		n, _ = n.Contiguous()
		x, _ = n.Reshape(1, (h/2)*(w/2), 4*C)
	}
	seq := len(ref.PromptEmbeds) / m.Cfg.JointDim
	pe, _ := tensor.New(ref.PromptEmbeds, []int{1, seq, m.Cfg.JointDim}, dev)
	pooled, _ := tensor.New(ref.Pooled, []int{1, m.Cfg.PooledDim}, dev)
	C, h, w := ref.LatentsShape[1], ref.LatentsShape[2], ref.LatentsShape[3]
	_ = C
	imgIDs := make([]float32, (h/2)*(w/2)*3)
	for r := range h / 2 {
		for c := range w / 2 {
			i := (r*(w/2) + c) * 3
			imgIDs[i+1], imgIDs[i+2] = float32(r), float32(c)
		}
	}
	cmp := func(name string, got *tensor.Tensor) {
		want, ok := tr[name]
		if !ok || got == nil {
			return
		}
		stats(t, name, got, tensor.MustSlice[float32](want))
	}
	cmpJoint := func(name string, h *tensor.Tensor) { // diffusers returns (text, image) halves
		txt, _ := h.Narrow(1, 0, seq)
		img, _ := h.Narrow(1, seq, h.Dim(1)-seq)
		txt, _ = txt.Contiguous()
		img, _ = img.Contiguous()
		cmp(name+".txt", txt)
		cmp(name+".img", img)
	}
	held := map[string]*tensor.Tensor{}
	models.SetFluxDebug(func(name string, x *tensor.Tensor) {
		cmp(name, x)
		switch name {
		case "q_rope", "k_rope", "v_in":
			held[name], _ = x.To(device.CPU())
		case "sdpa_out":
			if _, ok := held["sdpa_checked"]; ok {
				return
			}
			held["sdpa_checked"] = x
			checkSDPA(t, held["q_rope"], held["k_rope"], held["v_in"], x)
		}
		if name == "qn_img" || name == "w_norm_q" {
			c, _ := x.To(device.CPU())
			g := tensor.MustSlice[float32](c)
			t.Logf("%s light %v", name, g[:6])
			if want, ok := tr[name]; ok {
				t.Logf("%s ref   %v", name, tensor.MustSlice[float32](want)[:6])
			}
		}
	})
	defer models.SetFluxDebug(nil)
	m.Trace = func(kind string, i int, x, ctx *tensor.Tensor) error {
		switch {
		case kind == "double" && i == 0:
			cmp("double0.img", x)
			cmp("double0.txt", ctx)
			if stopAfter0 {
				return errStop
			}
		case kind == "double" && i == m.Cfg.NumLayers-1:
			cmp("double_last.img", x)
			cmp("double_last.txt", ctx)
		case kind == "single" && i == 0:
			cmpJoint("single0", x)
		case kind == "single" && i == m.Cfg.NumSingleLayers-1:
			cmpJoint("single_last", x)
		}
		return nil
	}
	y, err := m.Forward(models.FluxInputs{Image: x, Text: pe, Pooled: pooled, Timestep: []float32{ref.Sigmas[0]},
		ImgIDs: imgIDs, TxtIDs: make([]float32, seq*3)})
	if errors.Is(err, errStop) {
		return
	}
	if err != nil {
		t.Fatal(err)
	}
	cmp("out", y)
}

// checkSDPA recomputes a few query rows of softmax(q·kᵀ/√D)·v in float64
// from the (B, H, S, D) inputs and compares them with out.
func checkSDPA(t *testing.T, q, k, v, out *tensor.Tensor) {
	t.Helper()
	if q == nil || k == nil || v == nil {
		return
	}
	B, H, S, D := q.Dim(0), q.Dim(1), q.Dim(2), q.Dim(3)
	_ = B
	qs, ks, vs := tensor.MustSlice[float32](q), tensor.MustSlice[float32](k), tensor.MustSlice[float32](v)
	oc, _ := out.To(device.CPU())
	os_ := tensor.MustSlice[float32](oc)
	inv := 1 / math.Sqrt(float64(D))
	var md, mx float64
	rows := []int{0, 1, 255, 256, 257, 700, S - 1}
	for h := 0; h < H; h += 5 {
		for _, i := range rows {
			logits := make([]float64, S)
			m := math.Inf(-1)
			for j := range S {
				var dot float64
				for d := range D {
					dot += float64(qs[(h*S+i)*D+d]) * float64(ks[(h*S+j)*D+d])
				}
				logits[j] = dot * inv
				m = math.Max(m, logits[j])
			}
			var sum float64
			for j := range S {
				logits[j] = math.Exp(logits[j] - m)
				sum += logits[j]
			}
			for d := range D {
				var acc float64
				for j := range S {
					acc += logits[j] / sum * float64(vs[(h*S+j)*D+d])
				}
				got := float64(os_[(h*S+i)*D+d])
				md = math.Max(md, math.Abs(got-acc))
				mx = math.Max(mx, math.Abs(acc))
			}
		}
	}
	t.Logf("SDPA on the real q / k / v (heads 0, 5, .., rows %v): max |Δ| %.3g vs float64 (max |o| %.3g)", rows, md, mx)
}
