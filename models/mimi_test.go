package models

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"light/audio"
	"light/device"
	"light/tensor"
)

// kyutai/mimi on testdata/mimi/front_center_24k.wav (1.43 s). Reference:
// scripts/verify/ref_mimi.py (transformers MimiModel, F32, eager, CUDA).
const mimiRepo = "kyutai/mimi"

var mimiRefCodes = [][]int64{
	{1698, 698, 1663, 1292, 414, 1640, 1331, 666, 84, 48, 269, 252, 1109, 268, 442, 774, 1677, 1511},
	{754, 1237, 1351, 651, 887, 1546, 1700, 243, 243, 1646, 1486, 920, 932, 520, 1592, 1752, 1234, 1269},
	{1079, 1055, 1523, 762, 161, 466, 1559, 783, 783, 323, 211, 1512, 873, 513, 972, 1900, 1411, 742},
	{489, 1667, 865, 885, 104, 1216, 546, 290, 290, 164, 508, 567, 61, 411, 24, 654, 1188, 798},
	{2025, 1528, 2014, 610, 1514, 469, 267, 1736, 1736, 1514, 1563, 931, 1827, 414, 1516, 891, 51, 420},
	{1447, 1299, 480, 601, 1739, 878, 555, 555, 1443, 11, 508, 1407, 2037, 1981, 820, 1137, 1356, 347},
	{1633, 2009, 1885, 1421, 927, 1723, 1978, 666, 1978, 1577, 1013, 1207, 1251, 927, 2043, 1668, 1630, 160},
	{948, 98, 300, 1125, 13, 908, 1744, 1744, 1744, 999, 298, 596, 1581, 1332, 846, 1448, 977, 13},
}

var mimiRefDecoded = []float32{0.011391, 0.012401, -0.001715, 0.001666, 0.007481, 0.00708, 0.00596, 0.004186}

const (
	mimiRefSamples = 34274
	mimiRefDecLen  = 34560
	mimiRefRMS     = 0.067377
	mimiRefMaxAbs  = 0.445970
)

func mimiLoad(t *testing.T, dev device.Device) *Mimi {
	t.Helper()
	home, _ := os.UserHomeDir()
	if m, _ := filepath.Glob(filepath.Join(home, ".cache/huggingface/hub/models--kyutai--mimi/snapshots/*/model.safetensors")); len(m) == 0 {
		t.Skip(mimiRepo + " not in HF cache")
	}
	os.Setenv("HF_HUB_OFFLINE", "1")
	m, err := LoadMimiFromHF(mimiRepo, dev)
	if err != nil {
		t.Fatal(err)
	}
	return m
}

func mimiAudio(t *testing.T, dev device.Device) *tensor.Tensor {
	t.Helper()
	raw, err := os.ReadFile("../testdata/mimi/front_center_24k.wav")
	if err != nil {
		t.Skip(err)
	}
	samples, rate, err := audio.DecodeWAV(raw)
	if err != nil {
		t.Fatal(err)
	}
	if rate != 24000 || len(samples) != mimiRefSamples {
		t.Fatalf("wav: %d Hz, %d samples", rate, len(samples))
	}
	x, err := tensor.New(samples, []int{1, 1, len(samples)}, dev)
	if err != nil {
		t.Fatal(err)
	}
	return x
}

func mimiCheck(t *testing.T, m *Mimi, dev device.Device, tol float64) ([]int64, []float32) {
	t.Helper()
	x := mimiAudio(t, dev)
	codes, err := m.Encode(x, 8)
	if err != nil {
		t.Fatal(err)
	}
	if codes.Dim(0) != 1 || codes.Dim(1) != 8 || codes.Dim(2) != len(mimiRefCodes[0]) {
		t.Fatalf("codes shape %v", codes.Shape())
	}
	cc, _ := codes.To(device.CPU())
	got := tensor.MustSlice[int64](cc)
	T := len(mimiRefCodes[0])
	// Codebooks 0–3 must match exactly. Deeper acoustic codebooks quantise a
	// residual that has shrunk by orders of magnitude: at frame 13 of this
	// clip codebook 4's two nearest centroids are 5e-4 apart, Light picks
	// the exact nearest (958) while torch.cdist's matmul form rounds to 414,
	// and codebooks 5/7 follow. A few such flips are allowed; the decoded
	// audio must still match.
	bad := 0
	for q := range 8 {
		for i := range T {
			if got[q*T+i] != mimiRefCodes[q][i] {
				bad++
				if q < 4 || bad > 4 {
					t.Errorf("codes[%d][%d] = %d, want %d", q, i, got[q*T+i], mimiRefCodes[q][i])
				}
			}
		}
	}
	dec, err := m.Decode(codes)
	if err != nil {
		t.Fatal(err)
	}
	if dec.Dim(2) != mimiRefDecLen {
		t.Fatalf("decoded %d samples, want %d", dec.Dim(2), mimiRefDecLen)
	}
	dc, _ := dec.To(device.CPU())
	y := tensor.MustSlice[float32](dc)
	if d := maxAbsDiff(y[:8], mimiRefDecoded); d > tol {
		t.Errorf("decoded[:8] max abs err %.2e > %.0e: %v", d, tol, y[:8])
	}
	var ss, mx float64
	for _, v := range y {
		ss += float64(v * v)
		mx = math.Max(mx, math.Abs(float64(v)))
	}
	rms := math.Sqrt(ss / float64(len(y)))
	t.Logf("codes mismatches %d/%d, decoded rms %.6f (ref %.6f) maxabs %.6f (ref %.6f)", bad, 8*T, rms, mimiRefRMS, mx, mimiRefMaxAbs)
	if math.Abs(rms-mimiRefRMS) > 1e-3 || math.Abs(mx-mimiRefMaxAbs) > 5e-3 {
		t.Errorf("decoded stats off: rms %.6f maxabs %.6f", rms, mx)
	}
	return got, y
}

func TestMimiMatchesTransformersCPU(t *testing.T) {
	m := mimiLoad(t, device.CPU())
	t.Logf("params %.1f M", float64(m.ParamCount())/1e6)
	mimiCheck(t, m, device.CPU(), 1e-3)
}

func TestMimiMatchesTransformersCUDA(t *testing.T) {
	if !device.CUDAAvailable() {
		t.Skip("no CUDA")
	}
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	m := mimiLoad(t, gpu)
	mimiCheck(t, m, gpu, 1e-3)
}
