package splat

import (
	"image/color"
	"testing"
)

// One red Gaussian at the origin seen head-on covers the image centre, not
// the corners; behind it a blue one is hidden where they overlap.
func TestRenderOcclusionAndColour(t *testing.T) {
	c := float32((1 - 0.5) / shC0) // f_dc for a channel value of 1
	z := float32(-0.5 / shC0)      // ... of 0
	g := &Gaussians{
		XYZ:        []float32{0, 0, 0, 0, -0.3, 0}, // yaw 0 puts the camera on +y: the blue one is behind
		FeaturesDC: []float32{c, z, z, z, z, c},
		Scaling:    []float32{0.1, 0.1, 0.1, 0.2, 0.2, 0.2},
		Rotation:   []float32{1, 0, 0, 0, 1, 0, 0, 0},
		Opacity:    []float32{0.99, 0.99},
	}
	img := Render(g, Camera{Yaw: 0, Pitch: 0, Radius: 2, FOV: 40}, 64, 64, color.NRGBA{255, 255, 255, 255})
	at := func(x, y int) [3]uint8 {
		o := 4 * (y*64 + x)
		return [3]uint8{img.Pix[o], img.Pix[o+1], img.Pix[o+2]}
	}
	if p := at(32, 32); p[0] < 200 || p[2] > 60 {
		t.Errorf("centre %v, want red in front", p)
	}
	if p := at(1, 1); p != [3]uint8{255, 255, 255} {
		t.Errorf("corner %v, want background", p)
	}
}
