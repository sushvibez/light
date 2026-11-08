package models

import (
	"os"
	"runtime"
	"strconv"
	"testing"
	"time"

	"light/device"
	"light/tensor"
)

// TestMoshikoProfile times the pieces of one conversational frame on the
// real checkpoint (gated): Mimi encode, LM step, depformer, Mimi decode.
func TestMoshikoProfile(t *testing.T) {
	if os.Getenv("LIGHT_HEAVY_TESTS") == "" {
		t.Skip("set LIGHT_HEAVY_TESTS=1")
	}
	path := moshiGGUFPath()
	if path == "" || !device.CUDAAvailable() {
		t.Skip("moshiko / CUDA unavailable")
	}
	dev, _ := device.CUDA(0)
	os.Setenv("HF_HUB_OFFLINE", "1")
	mimi, err := LoadMimiFromHF("kyutai/mimi", dev)
	if err != nil {
		t.Skip(err)
	}
	m, err := LoadMoshiGGUF(path, dev)
	if err != nil {
		t.Fatal(err)
	}
	audio := mimiAudio(t, dev)
	n := 17
	if v := os.Getenv("LIGHT_MOSHI_FRAMES"); v != "" {
		n, _ = strconv.Atoi(v)
	}
	gc := os.Getenv("LIGHT_MOSHI_GC") == "1" // as MoshiGen.Step with GCEvery 1
	tEnc, tLM, tDep, tDec, tGC := moshiFrameLoop(t, m, mimi, audio, n, gc)
	f := float64(n)
	t.Logf("per frame: mimi encode %.1f ms, LM forward %.1f ms, depformer %.1f ms, mimi decode %.1f ms, gc %.1f ms, total %.1f ms (n=%d, gc %v)",
		ms(tEnc)/f, ms(tLM)/f, ms(tDep)/f, ms(tDec)/f, ms(tGC)/f, ms(tEnc+tLM+tDep+tDec+tGC)/f, n, gc)
}

// moshiFrameLoop runs n frames (cycling through the clip) and returns the
// time spent in each stage; kept as its own function so pprof -focus can
// isolate it from model loading.
func moshiFrameLoop(t *testing.T, m *Moshi, mimi *Mimi, audio *tensor.Tensor, n int, gc bool) (tEnc, tLM, tDep, tDec, tGC time.Duration) {
	dev := audio.Device()
	hop := mimi.Cfg.HopLength()
	frames := audio.Dim(2) / hop
	codec := mimi.NewStream()
	g := NewMoshiGen(m, nil, nil)
	prev := m.Cfg.TextStartToken()
	sync := func() { x, _ := tensor.New([]float32{1}, []int{1}, dev); _, _ = x.To(device.CPU()) }
	for step := 0; step < n; step++ {
		if gc {
			tg := time.Now()
			runtime.GC()
			tGC += time.Since(tg)
		}
		chunk, _ := audio.Narrow(2, (step%frames)*hop, hop)
		chunk, _ = chunk.Contiguous()
		t0 := time.Now()
		codes, err := codec.EncodeStep(chunk, 8)
		if err != nil {
			t.Fatal(err)
		}
		host, _ := codes.To(device.CPU())
		user := append([]int64(nil), tensor.MustSlice[int64](host)...)
		tEnc += time.Since(t0)
		// LM step split: Step does forward + depformer; time them apart.
		t1 := time.Now()
		audioT := make([]*tensor.Tensor, m.Cfg.AudioCodebooks)
		for cb := range audioT {
			audioT[cb], _ = tensor.New([]int64{int64(cb % 2000)}, []int{1, 1}, dev)
		}
		text, _ := tensor.New([]int64{prev}, []int{1, 1}, dev)
		logits, ys, err := m.Forward(text, audioT, g.Cache)
		if err != nil {
			t.Fatal(err)
		}
		row, _ := rowF32(logits)
		tLM += time.Since(t1)
		t2 := time.Now()
		if _, _, err := m.DepformerStep(ys, int64(Argmax(row)), nil, Argmax, g.DepCache); err != nil {
			t.Fatal(err)
		}
		tDep += time.Since(t2)
		t3 := time.Now()
		own, _ := tensor.New(user[:8], []int{1, 8, 1}, dev)
		pcm, err := codec.DecodeStep(own)
		if err != nil {
			t.Fatal(err)
		}
		_, _ = pcm.To(device.CPU())
		sync()
		tDec += time.Since(t3)
		prev = int64(Argmax(row))
	}
	return
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }
