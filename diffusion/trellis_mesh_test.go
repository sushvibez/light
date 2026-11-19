package diffusion

import (
	"fmt"
	"math"
	"path/filepath"
	"testing"

	"light/device"
	"light/mesh"
	"light/models"
	"light/safetensors"
	"light/tensor"
)

// The tiny pipeline's mesh decoder (scripts/verify/ref_trellis_mesh.py) on
// the stage-2 reference SLat (1186 voxels, 75,904 cubes at 64³): vertex and
// face counts and the sums of the vertices, colours and face indices
// TRELLIS's SLatMeshDecoder gives; an OBJ round trip of its size; and
// Generate's mesh.
func TestTrellisPipelineTinyMesh(t *testing.T) {
	ref, _, err := safetensors.LoadFiles([]string{"../models/testdata/trellis-tiny-mesh.safetensors",
		"../models/testdata/trellis-tiny-slat.safetensors"}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	counts := tensor.MustSlice[int32](ref["pipe.counts"])
	moments := tensor.MustSlice[float32](ref["pipe.moments"])
	devs := []device.Device{device.CPU()}
	if g, err := device.WebGPU(0); err == nil {
		devs = append(devs, g)
	}
	for _, dev := range devs {
		t.Run(fmt.Sprint(dev), func(t *testing.T) {
			p, err := LoadTrellisPipeline(trellisTiny, dev, TrellisLoadOptions{DINOv2: trellisTiny + "/dinov2", Mesh: true})
			if err != nil {
				t.Fatal(err)
			}
			cv := tensor.MustSlice[int32](ref["sample.coords"])
			coords := make([][4]int32, len(cv)/4)
			for i := range coords {
				copy(coords[i][:], cv[4*i:4*i+4])
			}
			g, err := models.NewTrellisGrid(coords, dev)
			if err != nil {
				t.Fatal(err)
			}
			slat, err := ref["sample.slat"].To(dev)
			if err != nil {
				t.Fatal(err)
			}
			_, m, err := p.DecodeMesh(g, slat)
			if err != nil {
				t.Fatal(err)
			}
			t.Logf("%d vertices, %d faces (reference %d, %d)", m.NumVertices(), m.NumFaces(), counts[0], counts[1])
			if m.NumVertices() != int(counts[0]) || m.NumFaces() != int(counts[1]) {
				t.Fatal("counts differ")
			}
			got := make([]float64, 10)
			for i, v := range m.Vertices {
				got[i%3] += float64(v)
			}
			for i, c := range m.Colors {
				got[3+i%6] += float64(c)
			}
			var fs float64
			for _, f := range m.Faces {
				fs += float64(f)
			}
			if float32(fs) != moments[9] { // stored at F32
				t.Errorf("face index sum %v, reference %v", fs, moments[9])
			}
			for i := range 9 {
				d := math.Abs(got[i] - float64(moments[i]))
				t.Logf("sum %d: %.6g (reference %.6g)", i, got[i], moments[i])
				if d > 1e-3*math.Max(1, math.Abs(float64(moments[i]))) {
					t.Errorf("sum %d differs by %.3g", i, d)
				}
			}

			path := filepath.Join(t.TempDir(), "tiny.obj")
			if err := mesh.Write(path, m.YUp()); err != nil {
				t.Fatal(err)
			}

			res, err := p.Generate(TrellisOptions{Image: trellisReadPNG(t, trellisTiny+"/object.png"), Seed: 3, Steps: 4, SLatSteps: 4})
			if err != nil {
				t.Fatal(err)
			}
			if res.Mesh == nil || res.Mesh.NumFaces() == 0 {
				t.Fatal("Generate returned no mesh")
			}
			t.Logf("Generate: %d voxels, mesh %d vertices, %d faces", len(res.Coords), res.Mesh.NumVertices(), res.Mesh.NumFaces())
		})
	}
}
