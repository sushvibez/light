// Example 10 — build a synthetic GGUF file with a Q4_0 weight, load it, and
// dequantize it to F32.
//
// This mirrors the real llama.cpp load path at a miniature scale:
//   - construct a GGUF file in memory (header + metadata + one Q4_0 tensor)
//   - open the file, look up the tensor by name
//   - dequantize via the quant package — the kernels that real GGUF models
//     rely on for inference.
//
// Building and opening the file uses the new `light/gguf` package; the Q4_0
// block packing is done by hand so the example is self-contained.
package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"log"

	"light/device"
	"light/dtype"
	"light/gguf"
	"light/quant"
	"light/tensor"
)

// buildQ4_0 packs 32 signed 4-bit values in [-8, 7] plus an F16 scale into
// one 18-byte GGUF Q4_0 block.
func buildQ4_0(scale float32, values [32]int8) []byte {
	buf := make([]byte, 18)
	binary.LittleEndian.PutUint16(buf[0:2], uint16(dtype.HalfFromF32(scale)))
	for j := range 16 {
		lo := uint8(values[j]+8) & 0x0F
		hi := uint8(values[j+16]+8) & 0x0F
		buf[2+j] = lo | (hi << 4)
	}
	return buf
}

// byteReaderAt is an io.ReaderAt over a []byte.
type byteReaderAt struct{ b []byte }

func (r *byteReaderAt) ReadAt(p []byte, off int64) (int, error) {
	if off >= int64(len(r.b)) {
		return 0, nil
	}
	return copy(p, r.b[off:]), nil
}

func main() {
	// Step 1: forge a GGUF buffer holding one Q4_0 tensor + some metadata.
	var vals [32]int8
	for i := range vals {
		vals[i] = int8((i % 16) - 8) // -8..7 repeated
	}
	scale := float32(0.5)

	w := gguf.NewWriter()
	w.AddString("general.name", "example-10")
	w.AddString("general.architecture", "toy")
	w.AddU32("general.file_type", 2) // FILE_TYPE_MOSTLY_Q4_0 in llama.cpp's taxonomy
	if err := w.AddTensor("weight", []int{32}, 2 /* GGML_TYPE_Q4_0 */, buildQ4_0(scale, vals)); err != nil {
		log.Fatal(err)
	}
	raw, err := w.Bytes()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("synthetic GGUF file: %d bytes\n", len(raw))

	// Step 2: parse it back.
	f, err := gguf.Open(&byteReaderAt{b: raw})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("version=%d  metadata keys=%d  tensors=%d\n",
		f.Version, len(f.KeyOrder), len(f.Order))
	if name, ok := f.String("general.name"); ok {
		fmt.Printf("  general.name     = %q\n", name)
	}
	if arch, ok := f.String("general.architecture"); ok {
		fmt.Printf("  general.architecture = %q\n", arch)
	}
	for _, n := range f.Order {
		info := f.Tensors[n]
		fmt.Printf("  tensor %-10s dtype=%-5v dims=%v offset=%d\n", n, info.DType, info.Dims, info.Offset)
	}

	// Step 3: materialize the Q4_0 tensor on CPU, then dequantize to F32.
	qt, err := f.Tensor("weight", device.CPU())
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("\nloaded tensor: dtype=%v shape=%v\n", qt.DType(), qt.Shape().Dims())

	dq, err := quant.DequantQ4_0(qt)
	if err != nil {
		log.Fatal(err)
	}
	vals32 := tensor.MustSlice[float32](dq)

	fmt.Println("\ndequantized F32 values (first 16):")
	var line bytes.Buffer
	for i := range 16 {
		fmt.Fprintf(&line, "% 6.2f ", vals32[i])
	}
	fmt.Println(line.String())
	fmt.Println("expected (scale * int4):")
	line.Reset()
	for i := range 16 {
		fmt.Fprintf(&line, "% 6.2f ", scale*float32(vals[i]))
	}
	fmt.Println(line.String())
}
