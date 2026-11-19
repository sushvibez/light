package diffusion

import (
	"fmt"
	"image"
	"image/color"
	"math"
)

// IdolCropMargin is idol's crop margin per input kind: the square crop is
// the object's bounding box times this (renders tight, photos looser).
var IdolCropMargin = map[string]float64{"synthetic": 1.2, "real": 1.7}

// idolPlanes is an image as float planes in [0, 1]: R, G, B straight (not
// premultiplied) and A.
type idolPlanes struct {
	w, h       int
	r, g, b, a []float64
}

func idolPlanesOf(img image.Image) *idolPlanes {
	bd := img.Bounds()
	w, h := bd.Dx(), bd.Dy()
	p := &idolPlanes{w: w, h: h, r: make([]float64, w*h), g: make([]float64, w*h), b: make([]float64, w*h), a: make([]float64, w*h)}
	for y := range h {
		for x := range w {
			c := color.NRGBAModel.Convert(img.At(bd.Min.X+x, bd.Min.Y+y)).(color.NRGBA)
			i := y*w + x
			p.r[i], p.g[i], p.b[i], p.a[i] = float64(c.R)/255, float64(c.G)/255, float64(c.B)/255, float64(c.A)/255
		}
	}
	return p
}

// idolHasAlpha reports whether the image has a real alpha channel (some
// pixel not fully opaque), so no background removal is needed.
func idolHasAlpha(img image.Image) bool {
	bd := img.Bounds()
	for y := bd.Min.Y; y < bd.Max.Y; y++ {
		for x := bd.Min.X; x < bd.Max.X; x++ {
			if _, _, _, a := img.At(x, y).RGBA(); a < 0xffff {
				return true
			}
		}
	}
	return false
}

// lanczosWeights returns, for each output index, the first input index and
// the normalised Lanczos-3 weights (Pillow's resampling: the filter widens
// by the scale when shrinking; pixel centres at i + 0.5).
func lanczosWeights(in, out int) ([]int, [][]float64) {
	scale := float64(in) / float64(out)
	support := 3.0
	fs := max(scale, 1)
	support *= fs
	lanczos := func(x float64) float64 {
		if x == 0 {
			return 1
		}
		if x <= -3 || x >= 3 {
			return 0
		}
		px := math.Pi * x
		return 3 * math.Sin(px) * math.Sin(px/3) / (px * px)
	}
	starts := make([]int, out)
	ws := make([][]float64, out)
	for i := range out {
		c := (float64(i) + 0.5) * scale
		lo := max(int(c-support+0.5), 0)
		hi := min(int(c+support+0.5), in)
		w := make([]float64, hi-lo)
		var sum float64
		for j := range w {
			w[j] = lanczos((float64(lo+j) - c + 0.5) / fs)
			sum += w[j]
		}
		if sum != 0 {
			for j := range w {
				w[j] /= sum
			}
		}
		starts[i], ws[i] = lo, w
	}
	return starts, ws
}

// resize resamples to (ow, oh) with Lanczos-3, premultiplied by alpha as
// Pillow does for RGBA.
func (p *idolPlanes) resize(ow, oh int) *idolPlanes {
	pre := func(c []float64) []float64 {
		o := make([]float64, len(c))
		for i := range c {
			o[i] = c[i] * p.a[i]
		}
		return o
	}
	planes := [][]float64{pre(p.r), pre(p.g), pre(p.b), p.a}
	hs, hw := lanczosWeights(p.w, ow)
	vs, vw := lanczosWeights(p.h, oh)
	out := make([][]float64, 4)
	for k, src := range planes {
		tmp := make([]float64, ow*p.h)
		for y := range p.h {
			for x := range ow {
				var s float64
				for j, wt := range hw[x] {
					s += src[y*p.w+hs[x]+j] * wt
				}
				tmp[y*ow+x] = s
			}
		}
		dst := make([]float64, ow*oh)
		for y := range oh {
			for x := range ow {
				var s float64
				for j, wt := range vw[y] {
					s += tmp[(vs[y]+j)*ow+x] * wt
				}
				dst[y*ow+x] = math.Min(math.Max(s, 0), 1)
			}
		}
		out[k] = dst
	}
	r := &idolPlanes{w: ow, h: oh, a: out[3], r: out[0], g: out[1], b: out[2]}
	for i, a := range r.a { // back to straight colour
		if a > 0 {
			r.r[i], r.g[i], r.b[i] = math.Min(r.r[i]/a, 1), math.Min(r.g[i]/a, 1), math.Min(r.b[i]/a, 1)
		} else {
			r.r[i], r.g[i], r.b[i] = 0, 0, 0
		}
	}
	return r
}

// idolMatte predicts an alpha matte (row-major uint8) for RGB planes (3, h,
// w) in [0, 1]: BiRefNet's remove_background.
type idolMatte func(rgb []float32, w, h int) ([]uint8, error)

// idolPreprocess is idol's preprocess_image: resize the short side to size;
// if the image has no real alpha channel, matte (BiRefNet) predicts one
// from the resized RGB (quantised to 8 bits as the SDK's PIL image is);
// erode the alpha by one pixel, crop a square of margin × the object's
// bounding box around its centre (outside the image is transparent),
// resize to size × size and composite on black. Returns the RGB planes (3,
// size, size) in [0, 1] and the image.
func idolPreprocess(img image.Image, size int, margin float64, matte idolMatte) ([]float32, *image.NRGBA, error) {
	hasAlpha := idolHasAlpha(img)
	if !hasAlpha && matte == nil {
		return nil, nil, fmt.Errorf("idol: the image has no transparent background and BiRefNet (birefnet.safetensors) is not loaded, so give an RGBA image with the object cut out")
	}
	p := idolPlanesOf(img)
	sc := float64(size) / float64(min(p.w, p.h))
	p = p.resize(max(1, int(math.Round(float64(p.w)*sc))), max(1, int(math.Round(float64(p.h)*sc))))
	if !hasAlpha {
		n := p.w * p.h
		rgb := make([]float32, 3*n)
		for i := range n {
			for c, pl := range [3][]float64{p.r, p.g, p.b} {
				pl[i] = math.Round(pl[i]*255) / 255
				rgb[c*n+i] = float32(pl[i])
			}
		}
		a, err := matte(rgb, p.w, p.h)
		if err != nil {
			return nil, nil, fmt.Errorf("idol: background removal: %w", err)
		}
		for i := range n {
			p.a[i] = float64(a[i]) / 255
		}
	}
	// MinFilter(3) on the alpha.
	er := make([]float64, len(p.a))
	for y := range p.h {
		for x := range p.w {
			m := 1.0
			for dy := -1; dy <= 1; dy++ {
				for dx := -1; dx <= 1; dx++ {
					yy, xx := min(max(y+dy, 0), p.h-1), min(max(x+dx, 0), p.w-1)
					m = math.Min(m, p.a[yy*p.w+xx])
				}
			}
			er[y*p.w+x] = m
		}
	}
	p.a = er
	left, top, right, bottom := p.w, p.h, -1, -1
	for y := range p.h {
		for x := range p.w {
			if math.Round(p.a[y*p.w+x]*255) > 0 {
				left, right = min(left, x), max(right, x)
				top, bottom = min(top, y), max(bottom, y)
			}
		}
	}
	if right < 0 {
		return nil, nil, fmt.Errorf("idol: no object found in the image (the alpha channel is empty)")
	}
	cx, cy := float64(left+right)/2, float64(top+bottom)/2
	half := float64(max(right-left, bottom-top)) / 2 * margin
	x0, y0, x1, y1 := int(cx-half), int(cy-half), int(cx+half), int(cy+half)
	cw, ch := x1-x0, y1-y0
	crop := &idolPlanes{w: cw, h: ch, r: make([]float64, cw*ch), g: make([]float64, cw*ch), b: make([]float64, cw*ch), a: make([]float64, cw*ch)}
	for y := range ch {
		for x := range cw {
			sx, sy := x0+x, y0+y
			if sx < 0 || sy < 0 || sx >= p.w || sy >= p.h {
				continue
			}
			i, j := y*cw+x, sy*p.w+sx
			crop.r[i], crop.g[i], crop.b[i], crop.a[i] = p.r[j], p.g[j], p.b[j], p.a[j]
		}
	}
	crop = crop.resize(size, size)
	px := make([]float32, 3*size*size)
	out := image.NewNRGBA(image.Rect(0, 0, size, size))
	q := func(v float64) uint8 { return uint8(math.Round(math.Min(math.Max(v, 0), 1) * 255)) }
	for i := range size * size {
		a := crop.a[i]
		r, g, b := q(crop.r[i]*a), q(crop.g[i]*a), q(crop.b[i]*a)
		px[i], px[size*size+i], px[2*size*size+i] = float32(r)/255, float32(g)/255, float32(b)/255
		out.Pix[4*i], out.Pix[4*i+1], out.Pix[4*i+2], out.Pix[4*i+3] = r, g, b, 255
	}
	return px, out, nil
}
