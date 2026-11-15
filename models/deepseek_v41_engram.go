package models

import (
	"encoding/json"
	"fmt"
	"math"
	"math/big"
	"math/bits"
	"os"
	"path/filepath"

	"light/dtype"
	"light/tensor"
)

// DeepSeek-V4.1's engram: each position is hashed with the tokens before it
// (2-gram .. max_ngram_size-gram over a compressed, normalised token id
// space), each n-gram into n_heads prime-sized bucket ranges of a large FP8
// table. The hashing reproduces engram.py's NgramHashState exactly: the
// bucket primes are consecutive primes from engram_vocab_size, and the
// per-layer multipliers come from numpy's default_rng(10007 · layer) —
// SeedSequence + PCG64 + Generator.integers, ported below.

// v41Engram holds the hash layout and the token map shared by every engram
// layer; each layer's table stays on the host (rows are gathered per call).
type v41Engram struct {
	layerIDs []int
	maxNgram int
	nHeads   int
	headDim  int
	primes   [][]int64 // [engram layer][(n-2)·nHeads + head]
	offsets  [][]int64 // the bucket range starts, same indexing
	mult     [][]int64 // [engram layer][lookback]
	tokenMap []int64
	padID    int64
}

func newV41Engram(cfg DeepseekV41Config, dir string) (*v41Engram, error) {
	e := &v41Engram{layerIDs: cfg.EngramLayerIDs, maxNgram: cfg.EngramMaxNgram, nHeads: cfg.EngramNHeads, headDim: cfg.EngramHeadDim}
	if e.maxNgram < 2 || e.nHeads < 1 || e.headDim < 1 {
		return nil, fmt.Errorf("deepseek_v41: engram max_ngram_size %d, n_heads %d, head_dim %d", e.maxNgram, e.nHeads, e.headDim)
	}
	raw, err := os.ReadFile(filepath.Join(dir, "engram_token_map.json"))
	if err != nil {
		return nil, fmt.Errorf("deepseek_v41: the engram layers need engram_token_map.json (the tokenizer's compressed id map; scripts/deepseek_v41_engram_map.py writes it): %w", err)
	}
	var tm struct {
		TokenMap []int64 `json:"token_map"`
		Size     int64   `json:"compressed_vocab_size"`
	}
	if err := json.Unmarshal(raw, &tm); err != nil {
		return nil, fmt.Errorf("deepseek_v41: engram_token_map.json: %w", err)
	}
	if tm.Size != cfg.EngramCompressedVocab {
		// every multiplier derives from this size: a mismatch rehashes the whole table
		return nil, fmt.Errorf("deepseek_v41: engram token map has %d compressed ids, config says %d", tm.Size, cfg.EngramCompressedVocab)
	}
	if cfg.EngramPadTokenID < 0 || cfg.EngramPadTokenID >= len(tm.TokenMap) {
		return nil, fmt.Errorf("deepseek_v41: engram pad id %d outside the %d-token map", cfg.EngramPadTokenID, len(tm.TokenMap))
	}
	e.tokenMap, e.padID = tm.TokenMap, tm.TokenMap[cfg.EngramPadTokenID]
	cand := cfg.EngramVocabSize
	for range e.layerIDs {
		var ps, offs []int64
		var off int64
		for range (e.maxNgram - 1) * e.nHeads {
			for !big.NewInt(cand).ProbablyPrime(0) { // exact below 2⁶⁴
				cand++
			}
			ps, offs = append(ps, cand), append(offs, off)
			off += cand
			cand++
		}
		e.primes, e.offsets = append(e.primes, ps), append(e.offsets, offs)
	}
	bound := uint64(math.MaxInt64) / uint64(cfg.EngramCompressedVocab) / 2
	for _, id := range e.layerIDs {
		v, err := numpyIntegers(uint64(10007*id), e.maxNgram, max(bound, 1))
		if err != nil {
			return nil, err
		}
		for i := range v {
			v[i] = v[i]*2 + 1
		}
		e.mult = append(e.mult, v)
	}
	return e, nil
}

// hashIDs returns, per engram layer, the table rows (B·S·cols, cols =
// (max_ngram_size - 1)·n_heads) of every position of tokens (B, S).
func (e *v41Engram) hashIDs(tokens []int64, B, S int) ([][]int64, error) {
	cols := (e.maxNgram - 1) * e.nHeads
	out := make([][]int64, len(e.layerIDs))
	for li := range out {
		out[li] = make([]int64, B*S*cols)
	}
	tok := make([]int64, e.maxNgram)
	for b := range B {
		for s := range S {
			for k := range tok {
				tok[k] = e.padID
				if s >= k {
					t := tokens[b*S+s-k]
					if t < 0 || int(t) >= len(e.tokenMap) {
						return nil, fmt.Errorf("deepseek_v41: token %d outside the engram token map (%d)", t, len(e.tokenMap))
					}
					tok[k] = e.tokenMap[t]
				}
			}
			for li := range e.layerIDs {
				m := e.mult[li]
				rolling := tok[0] * m[0]
				dst := out[li][(b*S+s)*cols:]
				for n := 1; n < e.maxNgram; n++ {
					rolling ^= tok[n] * m[n]
					for h := range e.nHeads {
						c := (n-1)*e.nHeads + h
						dst[c] = rolling%e.primes[li][c] + e.offsets[li][c]
					}
				}
			}
		}
	}
	return out, nil
}

// v41EngramRows gathers rows of an FP8 E4M3 table (R, hd) with per-row
// block scales (R, hd/blk): E8M0 bytes (U8) or float, dequantized to F32.
func v41EngramRows(table, scale *tensor.Tensor, ids []int64) ([]float32, error) {
	if table.Rank() != 2 || scale.Rank() != 2 || !table.IsContiguous() || !scale.IsContiguous() {
		return nil, fmt.Errorf("deepseek_v41: engram table %v / scale %v", table.Shape(), scale.Shape())
	}
	R, hd, nb := table.Dim(0), table.Dim(1), scale.Dim(1)
	if scale.Dim(0) != R || nb == 0 || hd%nb != 0 {
		return nil, fmt.Errorf("deepseek_v41: engram table %v with scale %v", table.Shape(), scale.Shape())
	}
	blk := hd / nb
	var sc func(i int) float32
	switch scale.DType() {
	case dtype.U8:
		sb := scale.Storage().CPUBytes()[scale.Offset():]
		sc = func(i int) float32 { return e8m0ToF32(sb[i]) }
	case dtype.F32:
		sv := tensor.MustSlice[float32](scale) // copies; fine for the small fixtures that store F32 scales
		sc = func(i int) float32 { return sv[i] }
	default:
		return nil, fmt.Errorf("deepseek_v41: engram scale dtype %v", scale.DType())
	}
	var row func(r, j int) float32
	switch table.DType() {
	case dtype.F8E4M3:
		tb := table.Storage().CPUBytes()[table.Offset():]
		row = func(r, j int) float32 { return dtype.Float8(tb[r*hd+j]).ToF32() }
	default:
		return nil, fmt.Errorf("deepseek_v41: engram table dtype %v (want F8_E4M3)", table.DType())
	}
	out := make([]float32, len(ids)*hd)
	for i, id := range ids {
		if id < 0 || id >= int64(R) {
			return nil, fmt.Errorf("deepseek_v41: engram row %d outside the %d-row table", id, R)
		}
		r := int(id)
		for j := range hd {
			out[i*hd+j] = row(r, j) * sc(r*nb+j/blk)
		}
	}
	return out, nil
}

// e8m0ToF32 decodes a UE8M0 scale byte: 2^(b - 127), 0xFF NaN.
func e8m0ToF32(b uint8) float32 {
	if b == 0xFF {
		return float32(math.NaN())
	}
	if b == 0 {
		return math.Float32frombits(1 << 22) // 2⁻¹²⁷, a subnormal
	}
	return math.Float32frombits(uint32(b) << 23)
}

// --- numpy's default_rng(seed).integers(0, high, n, dtype=int64) ---------------

// numpySeedSeq is numpy's SeedSequence(entropy).generate_state(n, uint64)
// for a non-negative integer entropy (no spawn key, pool size 4).
func numpySeedSeq(entropy uint64, n int) []uint64 {
	var ent []uint32
	for e := entropy; e > 0; e >>= 32 {
		ent = append(ent, uint32(e))
	}
	if len(ent) == 0 {
		ent = []uint32{0}
	}
	const pool = 4
	mixer := make([]uint32, pool)
	hc := uint32(0x43b0d7e5)
	hashmix := func(v uint32) uint32 {
		v ^= hc
		hc *= 0x931e8875
		v *= hc
		return v ^ v>>16
	}
	mix := func(x, y uint32) uint32 {
		r := 0xca01f9dd*x - 0x4973f715*y
		return r ^ r>>16
	}
	for i := range mixer {
		if i < len(ent) {
			mixer[i] = hashmix(ent[i])
		} else {
			mixer[i] = hashmix(0)
		}
	}
	for src := range mixer {
		for dst := range mixer {
			if src != dst {
				mixer[dst] = mix(mixer[dst], hashmix(mixer[src]))
			}
		}
	}
	for src := pool; src < len(ent); src++ {
		for dst := range mixer {
			mixer[dst] = mix(mixer[dst], hashmix(ent[src]))
		}
	}
	hb := uint32(0x8b51f9dd)
	w := make([]uint32, 2*n)
	for i := range w {
		v := mixer[i%pool] ^ hb
		hb *= 0x58f38ded
		v *= hb
		w[i] = v ^ v>>16
	}
	out := make([]uint64, n)
	for i := range out {
		out[i] = uint64(w[2*i]) | uint64(w[2*i+1])<<32
	}
	return out
}

// pcg64 is numpy's PCG64 (XSL-RR 128/64).
type pcg64 struct{ hi, lo, incHi, incLo uint64 }

const pcgMulHi, pcgMulLo = 2549297995355413924, 4865540595714422341

func (p *pcg64) step() {
	h, l := bits.Mul64(p.lo, pcgMulLo)
	h += p.hi*pcgMulLo + p.lo*pcgMulHi
	var c uint64
	p.lo, c = bits.Add64(l, p.incLo, 0)
	p.hi, _ = bits.Add64(h, p.incHi, c)
}

func newPCG64(seed uint64) *pcg64 {
	v := numpySeedSeq(seed, 4)
	p := &pcg64{}
	// inc = (initseq << 1) | 1; state = 0; step; state += initstate; step
	p.incHi, p.incLo = v[2]<<1|v[3]>>63, v[3]<<1|1
	p.step()
	var c uint64
	p.lo, c = bits.Add64(p.lo, v[1], 0)
	p.hi, _ = bits.Add64(p.hi, v[0], c)
	p.step()
	return p
}

func (p *pcg64) next() uint64 {
	p.step()
	return bits.RotateLeft64(p.hi^p.lo, -int(p.hi>>58))
}

// numpyIntegers is default_rng(seed).integers(0, high, n, dtype=np.int64)
// for high > 2³² (numpy's 64-bit Lemire bounded draw).
func numpyIntegers(seed uint64, n int, high uint64) ([]int64, error) {
	rng := high - 1
	if rng <= math.MaxUint32 {
		return nil, fmt.Errorf("deepseek_v41: numpy's 32-bit bounded draw (high %d) is not ported", high)
	}
	p := newPCG64(seed)
	excl := rng + 1
	out := make([]int64, n)
	for i := range out {
		hi, lo := bits.Mul64(p.next(), excl)
		if lo < excl {
			thr := (math.MaxUint64 - rng) % excl
			for lo < thr {
				hi, lo = bits.Mul64(p.next(), excl)
			}
		}
		out[i] = int64(hi)
	}
	return out, nil
}
