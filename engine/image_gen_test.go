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
