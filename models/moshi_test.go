package models

import (
	"math"
	"math/rand"
	"os"
	"testing"

	"light/audio"
	"light/device"
	"light/nn"
	"light/tensor"
	"light/tokenizer"
)

// tinyMoshiConfig keeps every structural feature of Moshi (fused QKV,
// interleaved RoPE, sliding context, gated MLP, depformer slices with a
// shared cache, text + 2 own + 2 user codebooks) at toy size.
func tinyMoshiConfig() MoshiConfig {
	return MoshiConfig{
		Transformer: MoshiTransformerConfig{Dim: 32, NumHeads: 4, NumLayers: 2, Hidden: 24, Context: 3, RoPE: true, MaxPeriod: 10000},
		Depformer:   MoshiTransformerConfig{Dim: 16, NumHeads: 2, NumLayers: 2, Hidden: 12, Context: 2, RoPE: false, MaxPeriod: 10000},
		TextInVocab: 11, TextOutVocab: 10, AudioVocab: 9, AudioCodebooks: 4, DepQ: 2,
	}
}

// newTinyMoshi builds the tiny model with deterministic random F32 weights
// (generated on the host, so every device gets identical values).
func newTinyMoshi(t *testing.T, dev device.Device, seed int64) *Moshi {
	t.Helper()
	m, err := NewMoshi(tinyMoshiConfig(), dev)
	if err != nil {
		t.Fatal(err)
	}
	rng := rand.New(rand.NewSource(seed))
	fill := func(dims []int, scale float64) *tensor.Tensor {
		n := 1
		for _, d := range dims {
			n *= d
		}
		data := make([]float32, n)
		for i := range data {
			data[i] = float32(rng.NormFloat64() * scale)
		}
		x, err := tensor.New(data, dims, device.CPU())
		if err != nil {
			t.Fatal(err)
		}
		if x, err = x.To(dev); err != nil {
			t.Fatal(err)
		}
		return x
	}
	linear := func(l *nn.Linear) {
		if err := l.SetWeight(fill([]int{l.OutFeatures(), l.InFeatures()}, 1/math.Sqrt(float64(l.InFeatures())))); err != nil {
			t.Fatal(err)
		}
	}
	norm := func(n *nn.RMSNorm) { n.Weight = fill(n.Weight.Shape().Dims(), 0.3) }
	transformer := func(tr *moshiTransformer) {
		for _, l := range tr.linears() {
			linear(l)
		}
		for _, n := range tr.norms() {
			n.Weight = fill(n.Weight.Shape().Dims(), 0.3)
		}
	}
	transformer(m.Transformer)
	m.TextEmb.Weight = fill(m.TextEmb.Weight.Shape().Dims(), 0.5)
	for _, e := range m.AudioEmb {
		e.Weight = fill(e.Weight.Shape().Dims(), 0.5)
	}
	norm(m.OutNorm)
	linear(m.TextLinear)
	for _, s := range m.Depformer {
		transformer(s.Transformer)
		s.Emb.Weight = fill(s.Emb.Weight.Shape().Dims(), 0.5)
		linear(s.LinearIn)
		linear(s.LinearOut)
	}
	return m
}

func moshiTinyInputs(t *testing.T, cfg MoshiConfig, T int, dev device.Device) (*tensor.Tensor, []*tensor.Tensor) {
	t.Helper()
	text := make([]int64, T)
	for i := range text {
		text[i] = int64((i*7 + 1) % cfg.TextInVocab)
	}
	tt, err := tensor.New(text, []int{1, T}, dev)
	if err != nil {
		t.Fatal(err)
	}
	audio := make([]*tensor.Tensor, cfg.AudioCodebooks)
	for k := range audio {
		codes := make([]int64, T)
		for i := range codes {
			codes[i] = int64((i*3 + k*5 + 2) % cfg.AudioVocab)
		}
		if audio[k], err = tensor.New(codes, []int{1, T}, dev); err != nil {
			t.Fatal(err)
		}
	}
	return tt, audio
}

// TestMoshiTinyConsistency checks that a full-sequence forward equals the
// same frames fed one at a time through the KV cache (which exercises the
// sliding context both through the mask and through cache trimming), and
// that the depformer is deterministic given the hidden state.
func TestMoshiTinyConsistency(t *testing.T) {
	cpu := device.CPU()
	m := newTinyMoshi(t, cpu, 1)
	cfg := m.Cfg
	const T = 7
	text, audio := moshiTinyInputs(t, cfg, T, cpu)
	full, hidden, err := m.Forward(text, audio, NewKVCache(cfg.Transformer.NumLayers))
	if err != nil {
		t.Fatal(err)
	}
	fullRow := tensor.MustSlice[float32](full)
	cache := NewKVCache(cfg.Transformer.NumLayers)
	for i := range T {
		tt, _ := text.Narrow(1, i, 1)
		aa := make([]*tensor.Tensor, len(audio))
		for k := range audio {
			aa[k], _ = audio[k].Narrow(1, i, 1)
		}
		step, _, err := m.Forward(tt, aa, cache)
		if err != nil {
			t.Fatal(err)
		}
		got, err := rowF32(step)
		if err != nil {
			t.Fatal(err)
		}
		want := fullRow[i*cfg.TextOutVocab : (i+1)*cfg.TextOutVocab]
		if d := maxAbsDiff(got, want); d > 1e-4 {
			t.Errorf("frame %d: stepwise vs full logits differ by %.2e", i, d)
		}
	}
	if cache.Len != T || cache.K[0].Dim(2) != cfg.Transformer.Context {
		t.Errorf("cache len %d, keys kept %d (context %d)", cache.Len, cache.K[0].Dim(2), cfg.Transformer.Context)
	}
	last, _ := hidden.Narrow(1, T-1, 1)
	dep := m.NewDepformerCache()
	codes1, rows1, err := m.DepformerStep(last, 3, nil, Argmax, dep)
	if err != nil {
		t.Fatal(err)
	}
	codes2, rows2, err := m.DepformerStep(last, 3, nil, Argmax, dep)
	if err != nil {
		t.Fatal(err)
	}
	for s := range codes1 {
		if codes1[s] != codes2[s] || maxAbsDiff(rows1[s], rows2[s]) != 0 || len(rows1[s]) != cfg.AudioVocab-1 {
			t.Errorf("depformer slice %d not deterministic: %d vs %d", s, codes1[s], codes2[s])
		}
	}
	// A forced token changes what the next slice sees but not slice 0.
	codes3, rows3, err := m.DepformerStep(last, 3, []int64{5, -1}, Argmax, dep)
	if err != nil {
		t.Fatal(err)
	}
	if codes3[0] != codes1[0] || maxAbsDiff(rows3[0], rows1[0]) != 0 {
		t.Errorf("forcing slice 0's successor changed slice 0")
	}
	if codes1[0] != 5 && maxAbsDiff(rows3[1], rows1[1]) == 0 {
		t.Errorf("forced token did not reach slice 1")
	}
	// The streaming protocol runs end to end.
	g := NewMoshiGen(m, nil, nil)
	prev := cfg.TextStartToken()
	for step := range 5 {
		user := []int64{int64(step % cfg.AudioVocab), int64((step + 3) % cfg.AudioVocab)}
		if prev, err = g.Step(prev, user); err != nil {
			t.Fatal(err)
		}
		if a := g.LastAudioTokens(); (step < g.AcousticDelay) != (a == nil) {
			t.Errorf("step %d: audio ready %v", step, a != nil)
		} else if a != nil && len(a) != cfg.DepQ {
			t.Errorf("step %d: %d audio tokens", step, len(a))
		}
	}
	if g.Steps() != 5 || len(g.TextTokens()) != 5 {
		t.Errorf("gen steps %d tokens %d", g.Steps(), len(g.TextTokens()))
	}
}

// TestMoshiTinyCUDAParity runs the tiny model on both devices.
func TestMoshiTinyCUDAParity(t *testing.T) {
	if !device.CUDAAvailable() {
		t.Skip("no CUDA")
	}
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	cpu := device.CPU()
	mc := newTinyMoshi(t, cpu, 1)
	mg := newTinyMoshi(t, gpu, 1)
	cfg := mc.Cfg
	const T = 6
	textC, audioC := moshiTinyInputs(t, cfg, T, cpu)
	textG, audioG := moshiTinyInputs(t, cfg, T, gpu)
	lc, hc, err := mc.Forward(textC, audioC, NewKVCache(cfg.Transformer.NumLayers))
	if err != nil {
		t.Fatal(err)
	}
	lg, hg, err := mg.Forward(textG, audioG, NewKVCache(cfg.Transformer.NumLayers))
	if err != nil {
		t.Fatal(err)
	}
	lgc, err := lg.To(cpu)
	if err != nil {
		t.Fatal(err)
	}
	if d := maxAbsDiff(tensor.MustSlice[float32](lc), tensor.MustSlice[float32](lgc)); d > 1e-4 {
		t.Errorf("CPU vs CUDA text logits differ by %.2e", d)
	}
	lastC, _ := hc.Narrow(1, T-1, 1)
	lastG, _ := hg.Narrow(1, T-1, 1)
	cc, rc, err := mc.DepformerStep(lastC, 2, []int64{-1, 4}, Argmax, mc.NewDepformerCache())
	if err != nil {
		t.Fatal(err)
	}
	cg, rg, err := mg.DepformerStep(lastG, 2, []int64{-1, 4}, Argmax, mg.NewDepformerCache())
	if err != nil {
		t.Fatal(err)
	}
	for s := range cc {
		if cc[s] != cg[s] {
			t.Errorf("depformer slice %d: CPU %d CUDA %d", s, cc[s], cg[s])
		}
		if d := maxAbsDiff(rc[s], rg[s]); d > 1e-4 {
			t.Errorf("depformer slice %d logits differ by %.2e", s, d)
		}
	}
}

// moshiRefStep is one frame of the reference run (scripts/verify/moshi-ref,
// Rust moshi-core on the same Q8 GGUF, CPU, argmax, LCG user codes).
type moshiRefStep struct {
	textIn int64
	top5   [5]int // text logits top-5 ids
	logits [5]float32
	audio  [8]int64 // depformer argmax codes
}

var moshiRefSteps = []moshiRefStep{
	{32000, [5]int{3, 0, 11725, 304, 534}, [5]float32{12.8901, 6.5024, 1.5334, 1.4427, 1.0867}, [8]int64{1049, 243, 260, 16, 846, 2025, 1986, 1640}},
	{3, [5]int{3, 0, 9805, 304, 14650}, [5]float32{20.7097, 11.8796, 4.3664, 2.7916, 2.7844}, [8]int64{127, 1700, 59, 76, 66, 1182, 1373, 1971}},
	{3, [5]int{3, 0, 11725, 11510, 667}, [5]float32{20.5405, 18.3047, 5.8617, 3.6016, 2.5618}, [8]int64{1880, 243, 457, 290, 267, 347, 192, 2008}},
	{3, [5]int{3, 0, 1152, 11725, 543}, [5]float32{24.9841, 24.3382, 5.8271, 5.6952, 4.7264}, [8]int64{972, 91, 457, 290, 1809, 199, 1941, 424}},
	{3, [5]int{0, 3, 543, 667, 11510}, [5]float32{21.4415, 18.0022, 7.7818, 6.2846, 5.1891}, [8]int64{972, 91, 457, 922, 1335, 1030, 1433, 1665}},
	{0, [5]int{11510, 2769, 387, 342, 11725}, [5]float32{18.9845, 10.7704, 10.4087, 10.0806, 9.8550}, [8]int64{420, 91, 1178, 390, 1335, 1537, 192, 424}},
}

// moshiLCG reproduces the reference's user-code generator.
func moshiLCG(seed *uint64) int64 {
	*seed = *seed*6364136223846793005 + 1442695040888963407
	return int64((*seed >> 33) % 2048)
}

// TestMoshikoMatchesCandleQ8 runs the streaming protocol on the real
// checkpoint and compares each frame's text top-5 and audio codes with the
// Rust moshi-core reference on the same GGUF. 8.2 GB of Q8 weights on
// CUDA; gated.
func TestMoshikoMatchesCandleQ8(t *testing.T) {
	if os.Getenv("LIGHT_HEAVY_TESTS") == "" {
		t.Skip("set LIGHT_HEAVY_TESTS=1 (8.2 GB Q8 checkpoint)")
	}
	path := moshiGGUFPath()
	if path == "" {
		t.Skip(MoshikoRepo + " not in HF cache")
	}
	if !device.CUDAAvailable() {
		t.Skip("no CUDA")
	}
	requireHostMemGB(t, 6)
	releaseModelMemory(t)
	dev, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	m, err := LoadMoshiGGUF(path, dev)
	if err != nil {
		t.Fatal(err)
	}
	c := m.Cfg
	if c.Transformer.NumLayers != 32 || c.AudioCodebooks != 16 || c.DepQ != 8 || c.Depformer.NumLayers != 6 ||
		c.Transformer.Hidden != 11264 || c.Depformer.Hidden != 2816 || c.TextInVocab != 32001 || c.AudioVocab != 2049 {
		t.Fatalf("config %+v", c)
	}
	t.Logf("params %.2f B", float64(m.ParamCount())/1e9)
	g := NewMoshiGen(m, Argmax, Argmax)
	seed := uint64(12345)
	prev := c.TextStartToken()
	if len(moshiRefSteps) == 0 { // smoke: run the protocol, report, no reference yet
		for step := range 4 {
			user := make([]int64, 8)
			for i := range user {
				user[i] = moshiLCG(&seed)
			}
			tok, err := g.Step(prev, user)
			if err != nil {
				t.Fatal(err)
			}
			got := top5(g.TextLogits)
			t.Logf("step %d text_in %d top5 %v logits %.3f %.3f %.3f audio %v", step, prev, got, g.TextLogits[got[0]], g.TextLogits[got[1]], g.TextLogits[got[2]], g.LastCodes)
			prev = tok
		}
		t.Skip("reference values pending")
	}
	for step, ref := range moshiRefSteps {
		user := make([]int64, 8)
		for i := range user {
			user[i] = moshiLCG(&seed)
		}
		if prev != ref.textIn {
			t.Fatalf("step %d: text in %d, reference fed %d", step, prev, ref.textIn)
		}
		tok, err := g.Step(prev, user)
		if err != nil {
			t.Fatal(err)
		}
		got := top5(g.TextLogits)
		for i := range 2 {
			if got[i] != ref.top5[i] {
				t.Errorf("step %d: text top-%d = %d (%.3f), want %d (%.3f)", step, i+1, got[i], g.TextLogits[got[i]], ref.top5[i], ref.logits[i])
			}
		}
		for i, id := range ref.top5 {
			if d := math.Abs(float64(g.TextLogits[id] - ref.logits[i])); d > 0.5 {
				t.Errorf("step %d: logit[%d] = %.3f, want %.3f", step, id, g.TextLogits[id], ref.logits[i])
			}
		}
		t.Logf("step %d text %d top5 %v audio %v", step, tok, got, g.LastCodes)
		// Candle's CPU path quantizes activations to Q8 for the dot products,
		// so logits differ by up to ~0.1; a code may flip where the reference's
		// choice sits within 0.25 of ours (observed once: step 1 slice 7,
		// 3.154 vs 2.993, during the pad-forced warm-up).
		for s := range 8 {
			if got := g.LastCodes[s]; got != ref.audio[s] {
				row := g.LastRows[s]
				if row[ref.audio[s]] < row[got]-0.25 {
					t.Errorf("step %d: depformer code %d = %d (%.3f), want %d (%.3f)", step, s, got, row[got], ref.audio[s], row[ref.audio[s]])
				} else {
					t.Logf("step %d: depformer code %d near-tie: %d (%.3f) vs reference %d (%.3f)", step, s, got, row[got], ref.audio[s], row[ref.audio[s]])
				}
			}
		}
		prev = tok
	}
}

// TestMoshikoConverse runs an offline speech-to-speech turn on the Whisper
// clip with greedy sampling: user audio → Mimi codes → 18 frames + 25 extra
// of silence → Moshi's codes → Mimi → audio, plus the decoded text. Set
// LIGHT_MOSHI_OUT=/path.wav to keep the reply.
func TestMoshikoConverse(t *testing.T) {
	if os.Getenv("LIGHT_HEAVY_TESTS") == "" {
		t.Skip("set LIGHT_HEAVY_TESTS=1 (8.2 GB Q8 checkpoint + Mimi)")
	}
	path := moshiGGUFPath()
	if path == "" {
		t.Skip(MoshikoRepo + " not in HF cache")
	}
	if !device.CUDAAvailable() {
		t.Skip("no CUDA")
	}
	requireHostMemGB(t, 6)
	releaseModelMemory(t)
	dev, err := device.CUDA(0)
	if err != nil {
		t.Fatal(err)
	}
	os.Setenv("HF_HUB_OFFLINE", "1")
	mimi, err := LoadMimiFromHF("kyutai/mimi", dev)
	if err != nil {
		t.Skip(err)
	}
	m, tokPath, err := LoadMoshiFromHF(MoshikoRepo, dev)
	if err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.LoadSentencePieceModel(tokPath)
	if err != nil {
		t.Fatal(err)
	}
	turn, err := m.Converse(mimi, mimiAudio(t, dev), 25, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if turn.Frames != 18+25 || len(turn.TextTokens) != turn.Frames || turn.Audio == nil {
		t.Fatalf("frames %d, text %d, audio %v", turn.Frames, len(turn.TextTokens), turn.Audio)
	}
	host, _ := turn.Audio.To(device.CPU())
	pcm := tensor.MustSlice[float32](host)
	wantLen := (turn.Frames - 2) * mimi.Cfg.HopLength() // acoustic delay drops the first 2 frames
	if len(pcm) != wantLen {
		t.Errorf("reply has %d samples, want %d", len(pcm), wantLen)
	}
	rms, peak := 0.0, 0.0
	for _, v := range pcm {
		rms += float64(v * v)
		peak = math.Max(peak, math.Abs(float64(v)))
	}
	rms = math.Sqrt(rms / float64(len(pcm)))
	t.Logf("text tokens %v", turn.TextTokens)
	t.Logf("text %q", tok.Decode(turn.Words()))
	t.Logf("reply %.2f s, rms %.4f, peak %.4f", float64(len(pcm))/24000, rms, peak)
	if peak == 0 {
		t.Errorf("silent reply")
	}
	if out := os.Getenv("LIGHT_MOSHI_OUT"); out != "" {
		f, err := os.Create(out)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		if err := audio.WriteWAV16(f, pcm, 24000); err != nil {
			t.Fatal(err)
		}
	}
}
