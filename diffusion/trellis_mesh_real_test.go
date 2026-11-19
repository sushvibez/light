package diffusion

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"light/device"
	"light/dtype"
	"light/mesh"
	"light/models"
	"light/ops"
	"light/safetensors"
	"light/tensor"
)

// TRELLIS-image-large's mesh decoder on its real weights against TRELLIS's
// own decode_slat(['mesh']) (scripts/colab/trellis_check.py records it in
// <work_dir>): out_layer's 101 values per 256³ cube from the reference SLat,
// SparseFeatures2Mesh on the reference's own outputs (the extraction alone:
// the same mesh up to F32 / GPU summation order), and the mesh from Light's
// outputs (vertex / face counts, symmetric nearest-vertex distances in
// cubes, colours of matched vertices). Writes <work_dir>/light.obj (y up).
// Runs when LIGHT_TRELLIS_DIR names that folder and its ref.safetensors has
// the mesh keys; on CUDA when built with it (LIGHT_TRELLIS_CPU=1 forces the
// CPU).
func TestTrellisMeshRealMatchesReference(t *testing.T) {
	dir := os.Getenv("LIGHT_TRELLIS_DIR")
	if dir == "" {
		t.Skip("LIGHT_TRELLIS_DIR not set")
	}
	ref, _, err := safetensors.LoadFiles([]string{filepath.Join(dir, "ref.safetensors")}, device.CPU())
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := ref["mesh_feats"]; !ok {
		t.Skip("ref.safetensors has no mesh keys (written before trellis_check.py recorded the mesh decoder)")
	}
	var meta struct {
		FP16     bool
		Backends map[string]string
	}
	if raw, err := os.ReadFile(filepath.Join(dir, "ref.json")); err == nil {
		if err := json.Unmarshal(raw, &meta); err != nil {
			t.Fatal(err)
		}
	}
	host := func(x *tensor.Tensor) []float32 {
		t.Helper()
		x, err := x.To(device.CPU())
		if err != nil {
			t.Fatal(err)
		}
		if x.DType() != dtype.F32 {
			if x, err = ops.Cast(x, dtype.F32); err != nil {
				t.Fatal(err)
			}
		}
		if x, err = x.Contiguous(); err != nil {
			t.Fatal(err)
		}
		return tensor.MustSlice[float32](x)
	}
	dev := device.CPU()
	if os.Getenv("LIGHT_TRELLIS_CPU") != "1" {
		if g, err := device.CUDA(0); err == nil {
			dev = g
		}
	}
	o := TrellisLoadOptions{Mesh: true}
	if d := filepath.Join(dir, "dinov2-with-registers-large"); trellisDirExists(d) {
		o.DINOv2 = d
	}
	p, err := LoadTrellisPipeline(filepath.Join(dir, "TRELLIS-image-large"), dev, o)
	if err != nil {
		t.Fatal(err)
	}
	tol := 1.0
	if meta.FP16 {
		tol = 10
	}
	cv := tensor.MustSlice[int32](ref["coords"])
	coords := make([][4]int32, len(cv)/4)
	for i := range coords {
		copy(coords[i][:], cv[4*i:4*i+4])
	}
	g, err := models.NewTrellisGrid(coords, dev)
	if err != nil {
		t.Fatal(err)
	}
	slat, err := ref["slat"].To(dev)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("device %v; %d voxels; reference fp16 %v, backends %v", dev, len(coords), meta.FP16, meta.Backends)

	feats, fine, err := p.MeshDecoder.Forward(g, slat)
	if err != nil {
		t.Fatal(err)
	}
	got, want := host(feats), host(ref["mesh_feats"])
	if len(got) != len(want) {
		t.Fatalf("decoder outputs: %d values, want %d", len(got), len(want))
	}
	md, mr, se, sw := 0.0, 0.0, 0.0, 0.0
	for i := range got {
		d := float64(got[i] - want[i])
		md = math.Max(md, math.Abs(d))
		mr = math.Max(mr, math.Abs(float64(want[i])))
		se += d * d
		sw += float64(want[i]) * float64(want[i])
	}
	t.Logf("decoder outputs (%d cubes): max diff %.3g (max |ref| %.3g, rel %.3g), rms rel %.3g", fine.N(), md, mr, md/mr, math.Sqrt(se/sw))
	if md > 2e-3*tol*math.Max(1, mr) {
		t.Errorf("decoder outputs differ by %.3g", md)
	}

	wantMesh := &mesh.Mesh{Vertices: host(ref["mesh_vertices"]), Faces: tensor.MustSlice[int32](ref["mesh_faces"]),
		Colors: host(ref["mesh_colors"]), ColorChannels: 6}
	cell := 1 / float64(p.MeshDecoder.MeshResolution())

	// The extraction alone, from the reference's outputs.
	alone, err := models.TrellisFeaturesToMesh(fine.Coords, want, p.MeshDecoder.MeshResolution(), p.MeshDecoder.Args.Rep.UseColor)
	if err != nil {
		t.Fatal(err)
	}
	trellisMeshMatch(t, "extraction from the reference outputs", alone, wantMesh, cell, 1e-4, 1e-3)

	m, err := p.MeshDecoder.Mesh(fine, feats)
	if err != nil {
		t.Fatal(err)
	}
	trellisMeshMatch(t, "mesh from Light's outputs", m, wantMesh, cell, 1e-3, 1e-2)
	out := filepath.Join(dir, "light.obj")
	if err := mesh.Write(out, m.YUp()); err != nil {
		t.Fatal(err)
	}
	t.Logf("wrote %s", out)
}

// trellisMeshMatch compares got with want: vertex and face counts (within
// countTol relative), and the distance from each vertex to the nearest one
// of the other mesh in cubes (cell), both ways: 99.9% must lie within
// distTol cubes. Matched vertices (within distTol) compare their colours.
func trellisMeshMatch(t *testing.T, name string, got, want *mesh.Mesh, cell, countTol, distTol float64) {
	t.Helper()
	gv, gf, wv, wf := got.NumVertices(), got.NumFaces(), want.NumVertices(), want.NumFaces()
	t.Logf("%s: %d vertices, %d faces (reference %d, %d)", name, gv, gf, wv, wf)
	if math.Abs(float64(gv-wv)) > countTol*float64(wv) || math.Abs(float64(gf-wf)) > countTol*float64(wf) {
		t.Errorf("%s: counts differ by more than %.2g%%", name, 100*countTol)
	}
	if gv == 0 || wv == 0 {
		t.Fatalf("%s: empty mesh", name)
	}
	if gv == wv && gf == wf {
		same := 0
		for i := range got.Faces {
			if got.Faces[i] == want.Faces[i] {
				same++
			}
		}
		t.Logf("%s: same counts; %d of %d face indices equal", name, same, len(got.Faces))
	}
	dist := func(a, b *mesh.Mesh) (d []float64, nn []int) {
		type key [3]int32
		bins := map[key][]int32{}
		cellOf := func(v []float32) key {
			return key{int32(math.Floor(float64(v[0]) / cell)), int32(math.Floor(float64(v[1]) / cell)), int32(math.Floor(float64(v[2]) / cell))}
		}
		for i := range b.NumVertices() {
			k := cellOf(b.Vertices[3*i : 3*i+3])
			bins[k] = append(bins[k], int32(i))
		}
		d, nn = make([]float64, a.NumVertices()), make([]int, a.NumVertices())
		for i := range a.NumVertices() {
			v := a.Vertices[3*i : 3*i+3]
			k := cellOf(v)
			best, bj := math.Inf(1), -1
			for o := range 27 {
				for _, j := range bins[key{k[0] + int32(o/9-1), k[1] + int32(o/3%3-1), k[2] + int32(o%3-1)}] {
					w := b.Vertices[3*j : 3*j+3]
					dx, dy, dz := float64(v[0]-w[0]), float64(v[1]-w[1]), float64(v[2]-w[2])
					if e := dx*dx + dy*dy + dz*dz; e < best {
						best, bj = e, int(j)
					}
				}
			}
			d[i], nn[i] = math.Sqrt(best)/cell, bj // +Inf: nothing within a cube
		}
		return d, nn
	}
	for _, dir := range []struct {
		label string
		a, b  *mesh.Mesh
	}{{"Light -> reference", got, want}, {"reference -> Light", want, got}} {
		d, nn := dist(dir.a, dir.b)
		s := append([]float64(nil), d...)
		sort.Float64s(s)
		far, mean, n := 0, 0.0, 0
		var cmax, csum float64
		var cn int
		for i, x := range d {
			if x > distTol {
				far++
			}
			if !math.IsInf(x, 0) {
				mean += x
				n++
			}
			if x <= distTol && dir.a.Colors != nil && dir.b.Colors != nil {
				for k := range 3 {
					c := math.Abs(float64(dir.a.Colors[6*i+k] - dir.b.Colors[6*nn[i]+k]))
					cmax = math.Max(cmax, c)
					csum += c
					cn++
				}
			}
		}
		t.Logf("%s %s: nearest-vertex distance in cubes median %.3g, p99.9 %.3g, max %.3g, mean %.3g; %d of %d beyond %.3g; matched colours max diff %.3g, mean %.3g",
			name, dir.label, s[len(s)/2], s[len(s)*999/1000], s[len(s)-1], mean/math.Max(1, float64(n)), far, len(d), distTol, cmax, csum/math.Max(1, float64(cn)))
		if float64(far) > 1e-3*float64(len(d)) {
			t.Errorf("%s %s: %d of %d vertices farther than %.3g cubes from the other mesh", name, dir.label, far, len(d), distTol)
		}
		if cn > 0 && csum/float64(cn) > 1e-3 {
			t.Errorf("%s %s: matched colours differ by %.3g on average", name, dir.label, csum/float64(cn))
		}
	}
}
