package models

import (
	"fmt"
	"math"

	"light/autograd"
	"light/device"
	"light/dtype"
	"light/nn"
	"light/ops"
	"light/tensor"
)

// WhisperConfig mirrors config.json of openai/whisper-* (tiny … large-v3).
type WhisperConfig struct {
	VocabSize       int
	NumMelBins      int // 80 (128 for large-v3)
	DModel          int
	EncoderLayers   int
	EncoderHeads    int
	EncoderFFN      int
	DecoderLayers   int
	DecoderHeads    int
	DecoderFFN      int
	MaxSourcePos    int // 1500
	MaxTargetPos    int // 448
	EOSToken        int // <|endoftext|> 50257
	SOTToken        int // <|startoftranscript|> 50258
	NoTimestamps    int // <|notimestamps|>
	SuppressTokens  []int
	BeginSuppress   []int
	LangToID        map[string]int // "en" → 50259
	TaskToID        map[string]int // "transcribe"/"translate"
	IsMultilingual  bool
	ActivationIsErf bool // "gelu" = exact erf GELU (all Whisper checkpoints)
}

// whisperAttn is one attention module; KProj has no bias in Whisper.
type whisperAttn struct {
	Q, K, V, O *nn.Linear
}

type whisperEncLayer struct {
	LN1  *nn.LayerNorm
	Attn *whisperAttn
	LN2  *nn.LayerNorm
	FC1  *nn.Linear
	FC2  *nn.Linear
}

type whisperDecLayer struct {
	LN1   *nn.LayerNorm
	Self  *whisperAttn
	LN2   *nn.LayerNorm
	Cross *whisperAttn
	LN3   *nn.LayerNorm
	FC1   *nn.Linear
	FC2   *nn.Linear
}

// Whisper is the encoder-decoder speech model. Build with LoadWhisperFromHF.
type Whisper struct {
	Cfg Whisperconfig

	Conv1W, Conv1B *tensor.Tensor // (D, mel, 3), (D)
	Conv2W, Conv2B *tensor.Tensor // (D, D, 3), (D)
	EncPos         *tensor.Tensor // (1500, D) sinusoids (stored in the checkpoint)
	EncLayers      []*whisperEncLayer
	EncNorm        *nn.LayerNorm

	TokEmb    *nn.Embedding // (V, D); tied LM head
	DecPos    *nn.Embedding // (448, D) learned
	DecLayers []*whisperDecLayer
	DecNorm   *nn.LayerNorm
	LMHead    *nn.Linear

	scale *tensor.Tensor // (1,) 1/sqrt(headDim)
	dev   device.Device
}

// Whisperconfig is an alias kept for the struct field above.
type Whisperconfig = WhisperConfig

// WhisperDecoderCache stores self-attention K/V per decoder layer and the
// cross-attention K/V projected once from the encoder output.
type WhisperDecoderCache struct {
	SelfK, SelfV   []*tensor.Tensor
	CrossK, CrossV []*tensor.Tensor
	Len            int
}

func newWhisperAttn(d int, dev device.Device) (*whisperAttn, error) {
	q, err := nn.NewLinearUninit(d, d, true, dtype.F32, dev)
	if err != nil {
		return nil, err
	}
	k, err := nn.NewLinearUninit(d, d, false, dtype.F32, dev)
	if err != nil {
		return nil, err
	}
	v, err := nn.NewLinearUninit(d, d, true, dtype.F32, dev)
	if err != nil {
		return nil, err
	}
	o, err := nn.NewLinearUninit(d, d, true, dtype.F32, dev)
	if err != nil {
		return nil, err
	}
	return &whisperAttn{Q: q, K: k, V: v, O: o}, nil
}

// NewWhisper allocates an uninitialised model (F32) on dev.
func NewWhisper(cfg WhisperConfig, dev device.Device) (*Whisper, error) {
	D := cfg.DModel
	if D%cfg.EncoderHeads != 0 || D%cfg.DecoderHeads != 0 {
		return nil, fmt.Errorf("whisper: d_model %d not divisible by heads", D)
	}
	m := &Whisper{Cfg: cfg, dev: dev}
	var err error
	z := func(dims ...int) *tensor.Tensor {
		if err != nil {
			return nil
		}
		var t *tensor.Tensor
		t, err = tensor.Zeros(dtype.F32, dims, dev)
		return t
	}
	m.Conv1W, m.Conv1B = z(D, cfg.NumMelBins, 3), z(D)
	m.Conv2W, m.Conv2B = z(D, D, 3), z(D)
	m.EncPos = z(cfg.MaxSourcePos, D)
	if err != nil {
		return nil, err
	}
	ln := func() (*nn.LayerNorm, error) { return nn.NewLayerNorm(D, 1e-5, dev) }
	lin := func(in, out int) (*nn.Linear, error) { return nn.NewLinearUninit(in, out, true, dtype.F32, dev) }
	for range cfg.EncoderLayers {
		l := &whisperEncLayer{}
		if l.LN1, err = ln(); err != nil {
			return nil, err
		}
		if l.Attn, err = newWhisperAttn(D, dev); err != nil {
			return nil, err
		}
		if l.LN2, err = ln(); err != nil {
			return nil, err
		}
		if l.FC1, err = lin(D, cfg.EncoderFFN); err != nil {
			return nil, err
		}
		if l.FC2, err = lin(cfg.EncoderFFN, D); err != nil {
			return nil, err
		}
		m.EncLayers = append(m.EncLayers, l)
	}
	if m.EncNorm, err = ln(); err != nil {
		return nil, err
	}
	if m.TokEmb, err = nn.NewEmbeddingUninit(cfg.VocabSize, D, dtype.F32, dev); err != nil {
		return nil, err
	}
	if m.DecPos, err = nn.NewEmbeddingUninit(cfg.MaxTargetPos, D, dtype.F32, dev); err != nil {
		return nil, err
	}
	for range cfg.DecoderLayers {
		l := &whisperDecLayer{}
		if l.LN1, err = ln(); err != nil {
			return nil, err
		}
		if l.Self, err = newWhisperAttn(D, dev); err != nil {
			return nil, err
		}
		if l.LN2, err = ln(); err != nil {
			return nil, err
		}
		if l.Cross, err = newWhisperAttn(D, dev); err != nil {
			return nil, err
		}
		if l.LN3, err = ln(); err != nil {
			return nil, err
		}
		if l.FC1, err = lin(D, cfg.DecoderFFN); err != nil {
			return nil, err
		}
		if l.FC2, err = lin(cfg.DecoderFFN, D); err != nil {
			return nil, err
		}
		m.DecLayers = append(m.DecLayers, l)
	}
	if m.DecNorm, err = ln(); err != nil {
		return nil, err
	}
	if m.LMHead, err = nn.NewLinearTied(m.TokEmb.Weight, false, dev); err != nil {
		return nil, err
	}
	hd := D / cfg.DecoderHeads // encoder and decoder share d_model/heads in every release
	if m.scale, err = tensor.New([]float32{float32(1 / math.Sqrt(float64(hd)))}, []int{1}, dev); err != nil {
		return nil, err
	}
	return m, nil
}

// Device returns the device the weights live on.
func (m *Whisper) Device() device.Device { return m.dev }

// ParamCount counts weights (LM head tied).
func (m *Whisper) ParamCount() int {
	n := 0
	for _, t := range m.parameters() {
		n += t.NumElements()
	}
	return n
}

func (m *Whisper) parameters() []*tensor.Tensor {
	ps := []*tensor.Tensor{m.Conv1W, m.Conv1B, m.Conv2W, m.Conv2B, m.EncPos, m.TokEmb.Weight, m.DecPos.Weight}
	attn := func(a *whisperAttn) {
		ps = append(ps, a.Q.W, a.Q.B, a.K.W, a.V.W, a.V.B, a.O.W, a.O.B)
	}
	for _, l := range m.EncLayers {
		ps = append(ps, l.LN1.Gamma, l.LN1.Beta, l.LN2.Gamma, l.LN2.Beta, l.FC1.W, l.FC1.B, l.FC2.W, l.FC2.B)
		attn(l.Attn)
	}
	ps = append(ps, m.EncNorm.Gamma, m.EncNorm.Beta)
	for _, l := range m.DecLayers {
		ps = append(ps, l.LN1.Gamma, l.LN1.Beta, l.LN2.Gamma, l.LN2.Beta, l.LN3.Gamma, l.LN3.Beta, l.FC1.W, l.FC1.B, l.FC2.W, l.FC2.B)
		attn(l.Self)
		attn(l.Cross)
	}
	return append(ps, m.DecNorm.Gamma, m.DecNorm.Beta)
}

func (m *Whisper) gelu(x *tensor.Tensor) (*tensor.Tensor, error) {
	if m.Cfg.ActivationIsErf {
		return ops.GeluErf(x)
	}
	return ops.Gelu(x)
}

// Encode runs the audio encoder on a log-mel tensor (B, nMels, 3000) F32
// and returns (B, 1500, D).
func (m *Whisper) Encode(mel *tensor.Tensor) (*tensor.Tensor, error) {
	if mel.Rank() != 3 || mel.Dim(1) != m.Cfg.NumMelBins {
		return nil, fmt.Errorf("Whisper.Encode: want (B, %d, T) mel, got %v", m.Cfg.NumMelBins, mel.Shape())
	}
	B := mel.Dim(0)
	D := m.Cfg.DModel
	x, err := ops.Conv1d(mel, m.Conv1W, m.Conv1B, 1, 1)
	if err != nil {
		return nil, err
	}
	if x, err = m.gelu(x); err != nil {
		return nil, err
	}
	if x, err = ops.Conv1d(x, m.Conv2W, m.Conv2B, 2, 1); err != nil {
		return nil, err
	}
	if x, err = m.gelu(x); err != nil {
		return nil, err
	}
	T := x.Dim(2)
	if T != m.Cfg.MaxSourcePos {
		return nil, fmt.Errorf("Whisper.Encode: %d frames after conv, expected %d (feed exactly 30 s = 3000 mel frames)", T, m.Cfg.MaxSourcePos)
	}
	if x, err = autograd.Permute(x, 0, 2, 1); err != nil { // (B, T, D)
		return nil, err
	}
	if x, err = autograd.Contiguous(x); err != nil {
		return nil, err
	}
	if x, err = ops.Add(x, m.EncPos); err != nil {
		return nil, err
	}
	for _, l := range m.EncLayers {
		h, err := l.LN1.Forward(x)
		if err != nil {
			return nil, err
		}
		if h, err = m.attend(l.Attn, h, h, B, T, T, m.Cfg.EncoderHeads, nil); err != nil {
			return nil, err
		}
		if x, err = ops.Add(x, h); err != nil {
			return nil, err
		}
		if h, err = l.LN2.Forward(x); err != nil {
			return nil, err
		}
		if h, err = m.mlp(l.FC1, l.FC2, h); err != nil {
			return nil, err
		}
		if x, err = ops.Add(x, h); err != nil {
			return nil, err
		}
	}
	_ = D
	return m.EncNorm.Forward(x)
}

func (m *Whisper) mlp(fc1, fc2 *nn.Linear, h *tensor.Tensor) (*tensor.Tensor, error) {
	h, err := fc1.Forward(h)
	if err != nil {
		return nil, err
	}
	if h, err = m.gelu(h); err != nil {
		return nil, err
	}
	return fc2.Forward(h)
}

// attend computes O(softmax(Q Kᵀ/√d + mask) V) with Q from xq (B, Tq, D)
// and K, V from xkv (B, Tk, D). mask is (Tq, Tk) additive or nil.
func (m *Whisper) attend(a *whisperAttn, xq, xkv *tensor.Tensor, B, Tq, Tk, NH int, mask *tensor.Tensor) (*tensor.Tensor, error) {
	k, err := a.K.Forward(xkv)
	if err != nil {
		return nil, err
	}
	v, err := a.V.Forward(xkv)
	if err != nil {
		return nil, err
	}
	return m.attendKV(a, xq, k, v, B, Tq, Tk, NH, mask)
}

// attendKV is attend with K and V already projected ((B, Tk, D) each).
func (m *Whisper) attendKV(a *whisperAttn, xq, k, v *tensor.Tensor, B, Tq, Tk, NH int, mask *tensor.Tensor) (*tensor.Tensor, error) {
	D := m.Cfg.DModel
	HD := D / NH
	q, err := a.Q.Forward(xq)
	if err != nil {
		return nil, err
	}
	if q, err = headsForward(q, B, Tq, NH, HD); err != nil {
		return nil, err
	}
	if k, err = headsForward(k, B, Tk, NH, HD); err != nil {
		return nil, err
	}
	if v, err = headsForward(v, B, Tk, NH, HD); err != nil {
		return nil, err
	}
	if q, err = autograd.Reshape(q, B*NH, Tq, HD); err != nil {
		return nil, err
	}
	if k, err = autograd.Reshape(k, B*NH, Tk, HD); err != nil {
		return nil, err
	}
	if v, err = autograd.Reshape(v, B*NH, Tk, HD); err != nil {
		return nil, err
	}
	kT, err := autograd.Transpose(k, 1, 2)
	if err != nil {
		return nil, err
	}
	if kT, err = autograd.Contiguous(kT); err != nil {
		return nil, err
	}
	s, err := autograd.MatMul(q, kT)
	if err != nil {
		return nil, err
	}
	if s, err = ops.Mul(s, m.scale); err != nil {
		return nil, err
	}
	if mask != nil {
		if s, err = ops.Add(s, mask); err != nil {
			return nil, err
		}
	}
	if s, err = ops.Softmax(s, -1); err != nil {
		return nil, err
	}
	ctx, err := autograd.MatMul(s, v)
	if err != nil {
		return nil, err
	}
	if ctx, err = autograd.Reshape(ctx, B, NH, Tq, HD); err != nil {
		return nil, err
	}
	if ctx, err = autograd.Permute(ctx, 0, 2, 1, 3); err != nil {
		return nil, err
	}
	if ctx, err = autograd.Contiguous(ctx); err != nil {
		return nil, err
	}
	if ctx, err = autograd.Reshape(ctx, B, Tq, D); err != nil {
		return nil, err
	}
	return a.O.Forward(ctx)
}

// NewDecoderCache prepares a cache for one encoder output (B, 1500, D):
// cross-attention K/V are projected here, once.
func (m *Whisper) NewDecoderCache(enc *tensor.Tensor) (*WhisperDecoderCache, error) {
	c := &WhisperDecoderCache{
		SelfK: make([]*tensor.Tensor, len(m.DecLayers)), SelfV: make([]*tensor.Tensor, len(m.DecLayers)),
		CrossK: make([]*tensor.Tensor, len(m.DecLayers)), CrossV: make([]*tensor.Tensor, len(m.DecLayers)),
	}
	for i, l := range m.DecLayers {
		k, err := l.Cross.K.Forward(enc)
		if err != nil {
			return nil, err
		}
		v, err := l.Cross.V.Forward(enc)
		if err != nil {
			return nil, err
		}
		c.CrossK[i], c.CrossV[i] = k, v
	}
	return c, nil
}

// Decode runs the decoder on tokens (B, T) I64 against enc (B, 1500, D),
// appending to cache, and returns F32 logits (B, T, V).
func (m *Whisper) Decode(tokens, enc *tensor.Tensor, cache *WhisperDecoderCache) (*tensor.Tensor, error) {
	if tokens.Rank() != 2 || tokens.DType() != dtype.I64 {
		return nil, fmt.Errorf("Whisper.Decode: tokens must be (B, T) I64")
	}
	if cache == nil {
		var err error
		if cache, err = m.NewDecoderCache(enc); err != nil {
			return nil, err
		}
	}
	B, T := tokens.Dim(0), tokens.Dim(1)
	prior := cache.Len
	if prior+T > m.Cfg.MaxTargetPos {
		return nil, fmt.Errorf("Whisper.Decode: %d tokens exceed max_target_positions %d", prior+T, m.Cfg.MaxTargetPos)
	}
	dev := tokens.Device()
	Tk := enc.Dim(1)
	NH := m.Cfg.DecoderHeads
	x, err := m.TokEmb.Forward(tokens)
	if err != nil {
		return nil, err
	}
	posData := make([]int64, T)
	for i := range posData {
		posData[i] = int64(prior + i)
	}
	posIdx, err := tensor.New(posData, []int{T}, dev)
	if err != nil {
		return nil, err
	}
	pos, err := m.DecPos.Forward(posIdx)
	if err != nil {
		return nil, err
	}
	if x, err = ops.Add(x, pos); err != nil {
		return nil, err
	}
	var mask *tensor.Tensor
	if T > 1 {
		if mask, err = causalMaskF32(T, prior, dev); err != nil {
			return nil, err
		}
	}
	for i, l := range m.DecLayers {
		h, err := l.LN1.Forward(x)
		if err != nil {
			return nil, err
		}
		k, err := l.Self.K.Forward(h)
		if err != nil {
			return nil, err
		}
		v, err := l.Self.V.Forward(h)
		if err != nil {
			return nil, err
		}
		if cache.SelfK[i] != nil {
			if k, err = ops.Concat([]*tensor.Tensor{cache.SelfK[i], k}, 1); err != nil {
				return nil, err
			}
			if v, err = ops.Concat([]*tensor.Tensor{cache.SelfV[i], v}, 1); err != nil {
				return nil, err
			}
		}
		cache.SelfK[i], cache.SelfV[i] = k, v
		if h, err = m.attendKV(l.Self, h, k, v, B, T, prior+T, NH, mask); err != nil {
			return nil, err
		}
		if x, err = ops.Add(x, h); err != nil {
			return nil, err
		}
		if h, err = l.LN2.Forward(x); err != nil {
			return nil, err
		}
		if h, err = m.attendKV(l.Cross, h, cache.CrossK[i], cache.CrossV[i], B, T, Tk, NH, nil); err != nil {
			return nil, err
		}
		if x, err = ops.Add(x, h); err != nil {
			return nil, err
		}
		if h, err = l.LN3.Forward(x); err != nil {
			return nil, err
		}
		if h, err = m.mlp(l.FC1, l.FC2, h); err != nil {
			return nil, err
		}
		if x, err = ops.Add(x, h); err != nil {
			return nil, err
		}
	}
	cache.Len = prior + T
	if x, err = m.DecNorm.Forward(x); err != nil {
		return nil, err
	}
	return m.LMHead.Forward(x)
}

// causalMaskF32 is the (T, prior+T) additive mask with -inf above the diagonal.
func causalMaskF32(T, prior int, dev device.Device) (*tensor.Tensor, error) {
	total := prior + T
	negInf := float32(math.Inf(-1))
	data := make([]float32, T*total)
	for i := range T {
		for j := prior + i + 1; j < total; j++ {
			data[i*total+j] = negInf
		}
	}
	return tensor.New(data, []int{T, total}, dev)
}

// TranscribeOpts controls Whisper.Transcribe.
type TranscribeOpts struct {
	Language  string // "en", "de", … ("" = let the model pick on multilingual checkpoints)
	Translate bool   // translate to English instead of transcribing
	MaxTokens int    // default 224 (half the context, as OpenAI does); 512 for Qwen3-ASR
	Prompt    string // Qwen3-ASR only: system-prompt context / hotwords to bias the transcript
}

// Prompt builds the decoder prefix for opts:
// <|startoftranscript|> [<|lang|>] <|transcribe|>|<|translate|> <|notimestamps|>.
// When Language is empty on a multilingual model, the language token is
// detected from the first decoder step (returned separately by Transcribe).
func (m *Whisper) Prompt(opts TranscribeOpts) ([]int64, error) {
	ids := []int64{int64(m.Cfg.SOTToken)}
	if m.Cfg.IsMultilingual {
		if opts.Language != "" {
			id, ok := m.Cfg.LangToID[opts.Language]
			if !ok {
				return nil, fmt.Errorf("whisper: unknown language %q", opts.Language)
			}
			ids = append(ids, int64(id))
		}
		task := "transcribe"
		if opts.Translate {
			task = "translate"
		}
		ids = append(ids, int64(m.Cfg.TaskToID[task]))
	}
	return append(ids, int64(m.Cfg.NoTimestamps)), nil
}

// Transcribe greedily decodes one 30-second log-mel chunk (1, nMels, 3000)
// and returns the generated token ids (without the prompt and EOS). Token
// suppression follows generation_config.json (suppress_tokens always,
// begin_suppress_tokens on the first generated token).
func (m *Whisper) Transcribe(mel *tensor.Tensor, opts TranscribeOpts) ([]int64, error) {
	enc, err := m.Encode(mel)
	if err != nil {
		return nil, err
	}
	return m.TranscribeEncoded(enc, opts)
}

// TranscribeEncoded is Transcribe on a precomputed encoder output.
func (m *Whisper) TranscribeEncoded(enc *tensor.Tensor, opts TranscribeOpts) ([]int64, error) {
	if opts.MaxTokens <= 0 {
		opts.MaxTokens = m.Cfg.MaxTargetPos / 2
	}
	dev := enc.Device()
	prompt, err := m.Prompt(opts)
	if err != nil {
		return nil, err
	}
	cache, err := m.NewDecoderCache(enc)
	if err != nil {
		return nil, err
	}
	detectLang := m.Cfg.IsMultilingual && opts.Language == ""
	if detectLang {
		// Run <|sot|> alone, pick the best language token, then continue.
		row, err := m.decodeRow(prompt[:1], enc, cache)
		if err != nil {
			return nil, err
		}
		best, bestV := -1, float32(math.Inf(-1))
		for _, id := range m.Cfg.LangToID {
			if row[id] > bestV {
				best, bestV = id, row[id]
			}
		}
		rest := append([]int64{int64(best)}, prompt[1:]...)
		prompt = append(prompt[:1], rest...)
		row, err = m.decodeRow(rest, enc, cache)
		if err != nil {
			return nil, err
		}
		return m.greedy(row, enc, cache, opts.MaxTokens, dev)
	}
	row, err := m.decodeRow(prompt, enc, cache)
	if err != nil {
		return nil, err
	}
	return m.greedy(row, enc, cache, opts.MaxTokens, dev)
}

func (m *Whisper) decodeRow(ids []int64, enc *tensor.Tensor, cache *WhisperDecoderCache) ([]float32, error) {
	x, err := tensor.New(ids, []int{1, len(ids)}, enc.Device())
	if err != nil {
		return nil, err
	}
	logits, err := m.Decode(x, enc, cache)
	if err != nil {
		return nil, err
	}
	cpu, err := logits.To(device.CPU())
	if err != nil {
		return nil, err
	}
	flat := tensor.MustSlice[float32](cpu)
	V := m.Cfg.VocabSize
	return flat[(len(ids)-1)*V : len(ids)*V], nil
}

func (m *Whisper) greedy(row []float32, enc *tensor.Tensor, cache *WhisperDecoderCache, maxTokens int, dev device.Device) ([]int64, error) {
	var out []int64
	negInf := float32(math.Inf(-1))
	for step := 0; step < maxTokens; step++ {
		for _, id := range m.Cfg.SuppressTokens {
			if id < len(row) {
				row[id] = negInf
			}
		}
		if step == 0 {
			for _, id := range m.Cfg.BeginSuppress {
				if id < len(row) {
					row[id] = negInf
				}
			}
		}
		best := argmax(row)
		if best == m.Cfg.EOSToken {
			break
		}
		out = append(out, int64(best))
		var err error
		if row, err = m.decodeRow([]int64{int64(best)}, enc, cache); err != nil {
			return out, err
		}
	}
	return out, nil
}
