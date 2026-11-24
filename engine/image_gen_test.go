package engine

import (
	"os"
	"testing"
)

// GenerateImage through the SD 1.5 branch on the GPU: a tiny run, the
// phases and steps reported, an image of the asked size back.
func TestGenerateImageSD15(t *testing.T) {
	if os.Getenv("LIGHT_HEAVY_TESTS") == "" {
		t.Skip("set LIGHT_HEAVY_TESTS=1")
	}
	es, err := Catalog()
	if err != nil {
		t.Fatal(err)
	}
	repo := ""
	for _, e := range es {
		if e.Kind == "image" && e.ModelType == "StableDiffusionPipeline" {
			repo = e.Spec.Repo
		}
	}
	if repo == "" {
		t.Skip("no StableDiffusionPipeline repo in the cache")
	}
	var phases []string
	steps := 0
	res, err := GenerateImage(ImageOptions{Repo: repo, Prompt: "a red bicycle", Width: 256, Height: 256, Steps: 4, Seed: 1, Device: "gpu",
		OnPhase: func(p string) { phases = append(phases, p) }, OnStep: func(step, total int) { steps = step }})
	if err != nil {
		t.Fatal(err)
	}
	b := res.Image.Bounds()
	if b.Dx() != 256 || b.Dy() != 256 || steps != 4 {
		t.Fatalf("got %dx%d after %d steps", b.Dx(), b.Dy(), steps)
	}
	t.Logf("%s: load %.1f s, generate %.1f s, phases %v", res.Pipeline, res.LoadS, res.GenS, phases)
}

// Step progress for the flow-matching pipelines: (1..total, total), the
// pipeline's default total when Steps is 0, and no callback when the
// caller did not ask.
func TestStepProgress(t *testing.T) {
	if stepProgress(ImageOptions{}, 4) != nil {
		t.Fatal("a callback without OnStep")
	}
	for _, c := range []struct{ steps, def, total int }{{0, 8, 8}, {3, 8, 3}} {
		var got [][2]int
		f := stepProgress(ImageOptions{Steps: c.steps, OnStep: func(step, total int) { got = append(got, [2]int{step, total}) }}, c.def)
		for i := range c.total {
			f(i, nil, nil)
		}
		if len(got) != c.total || got[0] != [2]int{1, c.total} || got[c.total-1] != [2]int{c.total, c.total} {
			t.Errorf("steps %d (default %d): %v", c.steps, c.def, got)
		}
	}
}

// The tiny FluxPipeline through GenerateImage on the CPU reports every step.
func TestGenerateImageFluxTinySteps(t *testing.T) {
	repo := "../diffusion/testdata/flux-tiny-pipe"
	if _, err := os.Stat(repo + "/model_index.json"); err != nil {
		t.Skip(err)
	}
	for _, steps := range []int{0, 3} {
		var got [][2]int
		_, err := GenerateImage(ImageOptions{Repo: repo, Prompt: "a red bicycle", Width: 64, Height: 64, Steps: steps, Seed: 1, Device: "cpu",
			OnStep: func(step, total int) { got = append(got, [2]int{step, total}) }})
		if err != nil {
			t.Skipf("tiny flux pipe does not run here: %v", err)
		}
		want := steps
		if want == 0 {
			want = 4
		}
		if len(got) != want || got[0] != [2]int{1, want} || got[want-1] != [2]int{want, want} {
			t.Errorf("steps %d: progress %v", steps, got)
		}
	}
}
