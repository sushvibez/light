package models

import (
	"fmt"
	"testing"

	"light/device"
	"light/dtype"
	"light/nn"
	"light/safetensors"
	"light/tensor"
)

// Tiny TRELLIS mesh decoder (scripts/verify/ref_trellis_mesh.py: TRELLIS's
// own SLatMeshDecoder, SparseFeatures2Mesh and FlexiCubes, seeded, F32 from
// F16-stored torso and upsampling) against the reference on 20 unsorted
// voxels: the torso, both subdivide blocks, out_layer, and the mesh
// (vertices, faces and colours in TRELLIS's order).
func TestTrellisMeshTinyMatchesReference(t *testing.T) {
	ref, _, err := safetensors.LoadFiles([]string{"testdata/trellis-tiny-mesh.safetensors"}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	devs := []device.Device{device.CPU()}
	if g, err := device.WebGPU(0); err == nil {
		devs = append(devs, g)
	}
	for _, dev := range devs {
		t.Run(fmt.Sprint(dev), func(t *testing.T) {
			dec, err := LoadTrellisSLatMeshDecoder("testdata/trellis-tiny/ckpts/slat_dec_mesh_tiny_fp16", dev)
			if err != nil {
				t.Fatal(err)
			}
			g, err := NewTrellisGrid(trellisCoords4(t, ref["mesh.coords"]), dev)
			if err != nil {
				t.Fatal(err)
			}
			x, err := ref["mesh.x"].To(dev)
			if err != nil {
				t.Fatal(err)
			}
			torso, up1, up2, feats, fine, err := dec.stages(g, x)
			if err != nil {
				t.Fatal(err)
			}
			trellisCmp(t, "torso", torso, trellisF32(t, ref["mesh.torso"]), 1e-4)
			trellisCmp(t, "subdivide block 1", up1, trellisF32(t, ref["mesh.up1"]), 1e-4)
			trellisCmp(t, "subdivide block 2", up2, trellisF32(t, ref["mesh.up2"]), 1e-4)
			trellisCmp(t, "decoder outputs", feats, trellisF32(t, ref["mesh.feats"]), 1e-4)
			for i, c := range trellisCoords4(t, ref["mesh.up2_coords"]) {
				if fine.Coords[i] != [3]int32{c[1], c[2], c[3]} {
					t.Fatalf("cube %d at %v, reference %v", i, fine.Coords[i], c)
				}
			}

			m, err := dec.Mesh(fine, feats)
			if err != nil {
				t.Fatal(err)
			}
			// The colours interpolate along edges by s·α weights: 1e-5 output
			// differences reach 3e-5 there.
			trellisMeshCmp(t, "mesh", m.Vertices, m.Faces, m.Colors, ref, "mesh.", 1e-4)

			// SparseFeatures2Mesh on the reference outputs: the extraction alone.
			wantFeats := trellisF32(t, ref["mesh.feats"])
			m, err = TrellisFeaturesToMesh(fine.Coords, wantFeats, dec.MeshResolution(), true)
			if err != nil {
				t.Fatal(err)
			}
			trellisMeshCmp(t, "mesh from the reference outputs", m.Vertices, m.Faces, m.Colors, ref, "mesh.", 1e-6)
		})
	}
}

// trellisMeshCmp compares a mesh with the reference's <prefix>vertices,
// faces and colors: same counts, identical faces, values within tol.
func trellisMeshCmp(t *testing.T, name string, verts []float32, faces []int32, colors []float32, ref map[string]*tensor.Tensor, prefix string, tol float64) {
	t.Helper()
	wf := tensor.MustSlice[int32](ref[prefix+"faces"])
	wv := trellisF32(t, ref[prefix+"vertices"])
	t.Logf("%s: %d vertices, %d faces (reference %d, %d)", name, len(verts)/3, len(faces)/3, len(wv)/3, len(wf)/3)
	if len(verts) != len(wv) || len(faces) != len(wf) {
		t.Fatalf("%s: counts differ", name)
	}
	diff := 0
	for i := range faces {
		if faces[i] != wf[i] {
			diff++
		}
	}
	if diff > 0 {
		t.Errorf("%s: %d of %d face indices differ", name, diff, len(faces))
	}
	host := func(v []float32, w int) *tensor.Tensor {
		x, err := tensor.New(v, []int{len(v) / w, w}, device.CPU())
		if err != nil {
			t.Fatal(err)
		}
		return x
	}
	trellisCmp(t, name+" vertices", host(verts, 3), wv, tol)
	trellisCmp(t, name+" colours", host(colors, 6), trellisF32(t, ref[prefix+"colors"]), tol)
}

// The mesh decoder with 16-bit weight storage (as `light trellis` loads
// it): the outputs stay close to the F32 run's.
func TestTrellisMeshTinyStorage16(t *testing.T) {
	ref, _, err := safetensors.LoadFiles([]string{"testdata/trellis-tiny-mesh.safetensors"}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	var dec *TrellisSLatMeshDecoder
	if err := nn.WithWeightStorage16(dtype.F16, func() (err error) {
		dec, err = LoadTrellisSLatMeshDecoder("testdata/trellis-tiny/ckpts/slat_dec_mesh_tiny_fp16", device.CPU())
		return
	}); err != nil {
		t.Fatal(err)
	}
	g, err := NewTrellisGrid(trellisCoords4(t, ref["mesh.coords"]), device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	feats, fine, err := dec.Forward(g, ref["mesh.x"])
	if err != nil {
		t.Fatal(err)
	}
	trellisCmp(t, "decoder outputs (16-bit storage)", feats, trellisF32(t, ref["mesh.feats"]), 5e-3)
	m, err := dec.Mesh(fine, feats)
	if err != nil {
		t.Fatal(err)
	}
	if m.NumFaces() == 0 {
		t.Fatal("no faces")
	}
	t.Logf("16-bit storage: %d vertices, %d faces (reference %d, %d)", m.NumVertices(), m.NumFaces(),
		ref["mesh.vertices"].Dim(0), ref["mesh.faces"].Dim(0))
}
