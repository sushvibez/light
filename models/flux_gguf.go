package models

import (
	"fmt"
	"os"
	"strings"

	"light/device"
	"light/dtype"
	"light/gguf"
	"light/nn"
	"light/quant"
	"light/safetensors"
	"light/tensor"
)

// GGUF Flux transformers (the city96 / ComfyUI-GGUF files): BFL module
// names, 2-D weights in ggml quantized formats (Q8_0, Q5_K, Q4_K_S, ...),
// norms and biases F32 / F16. They load through the BFL renamer like a
// single safetensors file; quantized weights stay quantized on the device
// (nn.Linear runs ops.MatMulQ: dequantize + GEMM for the image sequence,
// the fused mat-vec for one row) and fused qkv / linear1 split by rows at
// the byte level, since quantized blocks run along the input dimension.

// loadFluxGGUF reads the file's tensors (quantized as they are) under BFL
// names and renames them to diffusers names.
func loadFluxGGUF(path string) (map[string]*tensor.Tensor, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	g, err := gguf.Open(f)
	if err != nil {
		return nil, fmt.Errorf("flux %s: %w", path, err)
	}
	raw := make(map[string]*tensor.Tensor, len(g.Order))
	for _, name := range g.Order {
		t, err := g.TensorRowMajor(name, device.CPU())
		if err != nil {
			return nil, fmt.Errorf("flux %s: %s: %w", path, name, err)
		}
		raw[strings.TrimPrefix(name, "model.diffusion_model.")] = t
	}
	if _, bfl := raw["img_in.weight"]; !bfl {
		return raw, nil
	}
	return fluxFromBFL(raw)
}

// ggufShapes is the row-major shape of each tensor, for FluxConfigAt.
func ggufShapes(path string) (map[string][]int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	g, err := gguf.Open(f)
	if err != nil {
		return nil, err
	}
	shapes := make(map[string][]int, len(g.Tensors))
	for name, ti := range g.Tensors {
		dims := make([]int, len(ti.Dims))
		for i, d := range ti.Dims {
			dims[len(dims)-1-i] = d
		}
		shapes[strings.TrimPrefix(name, "model.diffusion_model.")] = dims
	}
	return shapes, nil
}

// splitRows splits t (N, K) along its rows into parts of the given sizes:
// views for plain dtypes, byte slices for quantized ones (a row is a whole
// number of blocks, so rows are contiguous byte ranges).
func splitRows(t *tensor.Tensor, sizes []int) ([]*tensor.Tensor, error) {
	total := 0
	for _, n := range sizes {
		total += n
	}
	if t.Rank() != 2 || t.Dim(0) != total {
		return nil, fmt.Errorf("split %v into rows %v", t.Shape(), sizes)
	}
	out := make([]*tensor.Tensor, len(sizes))
	off := 0
	if !t.DType().IsQuantized() {
		for i, n := range sizes {
			v, err := t.Narrow(0, off, n)
			if err != nil {
				return nil, err
			}
			out[i] = v
			off += n
		}
		return out, nil
	}
	K := t.Dim(1)
	rowBytes := t.DType().ByteSize(K)
	buf := t.Storage().CPUBytes()
	if buf == nil || len(buf) < total*rowBytes {
		return nil, fmt.Errorf("split %v: no host bytes", t.Shape())
	}
	for i, n := range sizes {
		v, err := tensor.FromBytes(buf[off*rowBytes:(off+n)*rowBytes], t.DType(), []int{n, K}, device.CPU())
		if err != nil {
			return nil, err
		}
		out[i] = v
		off += n
	}
	return out, nil
}

// widen returns t as F32 (dequantized, or cast) for small tensors that are
// rearranged on the host.
func widen(t *tensor.Tensor) (*tensor.Tensor, error) {
	if t.DType().IsQuantized() {
		return quant.Dequantize(t)
	}
	if t.DType() != dtype.F32 {
		return castCPU(t, dtype.F32)
	}
	return t, nil
}

// loadFluxLinear fills l from hf's name.weight / name.bias: a placeholder
// layer takes a quantized weight as it is (on dev) or gets an F32 weight
// allocated; a layer with a weight gets the values copied (cast to its
// dtype). keep leaves hf's entries in place (streamed loads).
func loadFluxLinear(l *nn.Linear, hf map[string]*tensor.Tensor, name string, dev device.Device, keep bool) error {
	cp := copyExactCast
	if keep {
		cp = copyExactCastKeep
	}
	if l.W == nil {
		src, ok := hf[name+".weight"]
		if !ok {
			return fmt.Errorf("flux: missing tensor %q", name+".weight")
		}
		if src.DType().IsQuantized() {
			w, err := src.To(dev)
			if err != nil {
				return fmt.Errorf("%s: %w", name, err)
			}
			if err := l.SetWeight(w); err != nil {
				return fmt.Errorf("%s: %w", name, err)
			}
			if !keep {
				delete(hf, name+".weight")
			}
		} else {
			wdt, _ := nn.WeightStorage(src.Dim(1))
			w, err := tensor.Zeros(wdt, src.Shape().Dims(), dev)
			if err != nil {
				return err
			}
			if err := l.SetWeight(w); err != nil {
				return fmt.Errorf("%s: %w", name, err)
			}
			if err := cp(l.W, hf, name+".weight"); err != nil {
				return err
			}
		}
	} else if err := cp(l.W, hf, name+".weight"); err != nil {
		return err
	}
	if l.B != nil {
		if _, ok := hf[name+".bias"]; !ok {
			return fmt.Errorf("flux: missing tensor %q", name+".bias")
		}
		return cp(l.B, hf, name+".bias")
	}
	return nil
}

// checkpointIsQuantized reports whether the 2-D weights are ggml-quantized.
func checkpointIsQuantized(hf map[string]*tensor.Tensor) bool {
	for name, t := range hf {
		if t.Rank() == 2 && strings.HasSuffix(name, ".weight") && t.DType().IsQuantized() {
			return true
		}
	}
	return false
}

var _ = safetensors.Header // keep the import set stable across edits
