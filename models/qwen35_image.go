package models

import (
	"fmt"
	"image"
	"math"
)

// Qwen35ImageConfig is the image processor's geometry (Qwen2-VL family).
type Qwen35ImageConfig struct {
	PatchSize, TemporalPatchSize, MergeSize int
	MinPixels, MaxPixels                    int
}

// ImageConfig is the processor geometry matching the vision tower
// (min / max pixels from Qwen3.5's preprocessor_config: 256², 4096²).
func (m *Qwen35Vision) ImageConfig() Qwen35ImageConfig {
	return Qwen35ImageConfig{PatchSize: m.Cfg.PatchSize, TemporalPatchSize: m.Cfg.TemporalPatchSize,
		MergeSize: m.Cfg.SpatialMergeSize, MinPixels: 65536, MaxPixels: 16777216}
}

// qwenSmartResize is the processor's smart_resize: both sides a multiple of
// factor, the area within [minPixels, maxPixels], the aspect ratio kept.
func qwenSmartResize(h, w, factor, minPixels, maxPixels int) (int, int, error) {
	if float64(max(h, w))/float64(min(h, w)) > 200 {
		return 0, 0, fmt.Errorf("image aspect ratio %d:%d is over 200", w, h)
	}
	round := func(v float64) int { return int(math.RoundToEven(v)) } // Python round()
	hb := round(float64(h)/float64(factor)) * factor
	wb := round(float64(w)/float64(factor)) * factor
	switch {
	case hb*wb > maxPixels:
		beta := math.Sqrt(float64(h*w) / float64(maxPixels))
		hb = max(factor, int(math.Floor(float64(h)/beta/float64(factor)))*factor)
		wb = max(factor, int(math.Floor(float64(w)/beta/float64(factor)))*factor)
	case hb*wb < minPixels:
		beta := math.Sqrt(float64(minPixels) / float64(h*w))
		hb = int(math.Ceil(float64(h)*beta/float64(factor))) * factor
		wb = int(math.Ceil(float64(w)*beta/float64(factor))) * factor
	}
	return hb, wb, nil
}

// rgbBytes is img as row-major RGB bytes.
func rgbBytes(img image.Image) ([]uint8, int, int) {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	out := make([]uint8, w*h*3)
	for y := range h {
		for x := range w {
			r, g, bl, _ := img.At(b.Min.X+x, b.Min.Y+y).RGBA()
			i := (y*w + x) * 3
			out[i], out[i+1], out[i+2] = uint8(r>>8), uint8(g>>8), uint8(bl>>8)
		}
	}
	return out, w, h
}

// Pillow's 8-bit resampling: bicubic (a = −0.5) with support scaled for
// downsampling (antialias), coefficients in 22-bit fixed point, each pass
// rounded and clipped to uint8 (Resample.c; torchvision's antialiased uint8
// bicubic follows it).
const pilPrecisionBits = 32 - 8 - 2

func pilBicubic(x float64) float64 {
	const a = -0.5
	x = math.Abs(x)
	switch {
	case x < 1:
		return ((a+2)*x-(a+3))*x*x + 1
	case x < 2:
		return (((x-5)*x+8)*x - 4) * a
	}
	return 0
}

// pilCoeffs returns, per output index, the first input index and the
// fixed-point weights.
func pilCoeffs(inSize, outSize int) ([]int, [][]int32) {
	scale := float64(inSize) / float64(outSize)
	fs := max(scale, 1)
	support := 2.0 * fs
	starts := make([]int, outSize)
	ks := make([][]int32, outSize)
	for xx := range outSize {
		center := (float64(xx) + 0.5) * scale
		ss := 1 / fs
		xmin := max(int(center-support+0.5), 0)
		xmax := min(int(center+support+0.5), inSize) - xmin
		w := make([]float64, xmax)
		ww := 0.0
		for x := range xmax {
			w[x] = pilBicubic((float64(x+xmin) - center + 0.5) * ss)
			ww += w[x]
		}
		k := make([]int32, xmax)
		for x := range xmax {
			v := w[x]
			if ww != 0 {
				v /= ww
			}
			if v < 0 {
				k[x] = int32(-0.5 + v*(1<<pilPrecisionBits))
			} else {
				k[x] = int32(0.5 + v*(1<<pilPrecisionBits))
			}
		}
		starts[xx], ks[xx] = xmin, k
	}
	return starts, ks
}

func pilClip8(v int32) uint8 {
	v >>= pilPrecisionBits
	if v < 0 {
		return 0
	}
	if v > 255 {
		return 255
	}
	return uint8(v)
}

// pilResizeRGB resizes row-major RGB bytes from (w, h) to (ow, oh):
// horizontal pass, then vertical, as Pillow does.
func pilResizeRGB(src []uint8, w, h, ow, oh int) []uint8 {
	cur, cw := src, w
	if ow != w {
		starts, ks := pilCoeffs(w, ow)
		out := make([]uint8, ow*h*3)
		for y := range h {
			for x := range ow {
				for c := range 3 {
					ss := int32(1) << (pilPrecisionBits - 1)
					for i, k := range ks[x] {
						ss += int32(cur[(y*cw+starts[x]+i)*3+c]) * k
					}
					out[(y*ow+x)*3+c] = pilClip8(ss)
				}
			}
		}
		cur, cw = out, ow
	}
	if oh != h {
		starts, ks := pilCoeffs(h, oh)
		out := make([]uint8, cw*oh*3)
		for y := range oh {
			for x := range cw {
				for c := range 3 {
					ss := int32(1) << (pilPrecisionBits - 1)
					for i, k := range ks[y] {
						ss += int32(cur[((starts[y]+i)*cw+x)*3+c]) * k
					}
					out[(y*cw+x)*3+c] = pilClip8(ss)
				}
			}
		}
		cur = out
	}
	return cur
}

// Qwen35ImagePatches preprocesses an image as the Qwen2-VL image processor
// does: smart-resize to a multiple of patch·merge (antialiased bicubic),
// normalise to (x − 127.5) / 127.5, and flatten into patches of
// (C, temporal, P, P) (the frame repeated temporally) in merge-block order.
// Returns the patches (N · C·T·P·P, row-major) and the grid (1, h, w) in
// patches.
func Qwen35ImagePatches(img image.Image, c Qwen35ImageConfig) ([]float32, [3]int, error) {
	rgb, w, h := rgbBytes(img)
	P, T, M := c.PatchSize, c.TemporalPatchSize, c.MergeSize
	rh, rw, err := qwenSmartResize(h, w, P*M, c.MinPixels, c.MaxPixels)
	if err != nil {
		return nil, [3]int{}, err
	}
	if rh != h || rw != w {
		rgb = pilResizeRGB(rgb, w, h, rw, rh)
	}
	gh, gw := rh/P, rw/P
	dim := 3 * T * P * P
	out := make([]float32, gh*gw*dim)
	n := 0
	for bh := range gh / M {
		for bw := range gw / M {
			for mh := range M {
				for mw := range M {
					py0, px0 := (bh*M+mh)*P, (bw*M+mw)*P
					o := out[n*dim : (n+1)*dim]
					for ch := range 3 {
						for t := range T {
							for y := range P {
								for x := range P {
									v := float32(rgb[((py0+y)*rw+px0+x)*3+ch])
									o[((ch*T+t)*P+y)*P+x] = (v - 127.5) / 127.5
								}
							}
						}
					}
					n++
				}
			}
		}
	}
	return out, [3]int{1, gh, gw}, nil
}

// ImageRGB returns img as row-major RGB bytes with its width and height.
func ImageRGB(img image.Image) ([]uint8, int, int) { return rgbBytes(img) }

// ResizeRGB resizes row-major RGB bytes with Pillow's antialiased bicubic.
func ResizeRGB(src []uint8, w, h, ow, oh int) []uint8 { return pilResizeRGB(src, w, h, ow, oh) }
