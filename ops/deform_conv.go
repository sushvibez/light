package ops

import (
	"fmt"
	"math"

	"light/device"
	"light/dtype"
	"light/internal/parallel"
	"light/tensor"
)

// DeformConv2dOptions are torchvision.ops.deform_conv2d's stride, padding
// and dilation (one offset group, one weight group).
type DeformConv2dOptions struct {
	Stride, Padding, Dilation [2]int
}

// DeformConv2d is torchvision.ops.deform_conv2d (deformable convolution
// v2 with a mask): each output position samples the input bilinearly at
// its kernel taps moved by offset (B, 2·kh·kw, Ho, Wo; (dy, dx) per tap,
// taps row-major), scales each sample by mask (B, kh·kw, Ho, Wo; nil: 1),
// and applies weight (O, C, kh, kw) and bias (O; nil: none). Samples outside
// the input read zero. F32. The columns are gathered on the host (other
// devices round-trip through it), then one GEMM per image on the input's
// device.
func DeformConv2d(x, offset, mask, weight, bias *tensor.Tensor, o DeformConv2dOptions) (*tensor.Tensor, error) {
	for i := range 2 {
		if o.Stride[i] == 0 {
			o.Stride[i] = 1
		}
		if o.Dilation[i] == 0 {
			o.Dilation[i] = 1
		}
	}
	if x.Rank() != 4 || weight.Rank() != 4 || offset.Rank() != 4 {
		return nil, fmt.Errorf("DeformConv2d: want x (B, C, H, W), offset (B, 2K, Ho, Wo), weight (O, C, kh, kw); got %v, %v, %v", x.Shape(), offset.Shape(), weight.Shape())
	}
	B, C, H, W := x.Dim(0), x.Dim(1), x.Dim(2), x.Dim(3)
	O, kh, kw := weight.Dim(0), weight.Dim(2), weight.Dim(3)
	if weight.Dim(1) != C {
		return nil, fmt.Errorf("DeformConv2d: weight has %d input channels, x %d", weight.Dim(1), C)
	}
	K := kh * kw
	Ho := (H+2*o.Padding[0]-o.Dilation[0]*(kh-1)-1)/o.Stride[0] + 1
	Wo := (W+2*o.Padding[1]-o.Dilation[1]*(kw-1)-1)/o.Stride[1] + 1
	if offset.Dim(0) != B || offset.Dim(1) != 2*K || offset.Dim(2) != Ho || offset.Dim(3) != Wo {
		return nil, fmt.Errorf("DeformConv2d: offset %v, want (%d, %d, %d, %d)", offset.Shape(), B, 2*K, Ho, Wo)
	}
	if mask != nil && (mask.Rank() != 4 || mask.Dim(0) != B || mask.Dim(1) != K || mask.Dim(2) != Ho || mask.Dim(3) != Wo) {
		return nil, fmt.Errorf("DeformConv2d: mask %v, want (%d, %d, %d, %d)", mask.Shape(), B, K, Ho, Wo)
	}
	dev := x.Device()
	host := func(t *tensor.Tensor) ([]float32, error) {
		if t == nil {
			return nil, nil
		}
		c, err := t.To(device.CPU())
		if err != nil {
			return nil, err
		}
		if c.DType() != dtype.F32 {
			if c, err = Cast(c, dtype.F32); err != nil {
				return nil, err
			}
		}
		if c, err = ensureContig(c); err != nil {
			return nil, err
		}
		return tensor.MustSlice[float32](c), nil
	}
	xs, err := host(x)
	if err != nil {
		return nil, err
	}
	offs, err := host(offset)
	if err != nil {
		return nil, err
	}
	ms, err := host(mask)
	if err != nil {
		return nil, err
	}
	// The weight as (O, C·K) on dev, the GEMM's left operand.
	wf, err := ensureContig(weight)
	if err != nil {
		return nil, err
	}
	if wf.DType() != dtype.F32 {
		if wf, err = Cast(wf, dtype.F32); err != nil {
			return nil, err
		}
	}
	if wf, err = wf.Reshape(O, C*K); err != nil {
		return nil, err
	}
	P := Ho * Wo
	outs := make([]*tensor.Tensor, B)
	for b := range B {
		cols := make([]float32, C*K*P) // row (c, ky, kx), column (ho, wo)
		xb := xs[b*C*H*W : (b+1)*C*H*W]
		ob := offs[b*2*K*P : (b+1)*2*K*P]
		var mb []float32
		if ms != nil {
			mb = ms[b*K*P : (b+1)*K*P]
		}
		parallel.For(K*P, 256, func(lo, hi int) {
			for kp := lo; kp < hi; kp++ {
				k, p := kp/P, kp%P
				ki, kj := k/kw, k%kw
				ho, wo := p/Wo, p%Wo
				y := float64(ho*o.Stride[0]-o.Padding[0]+ki*o.Dilation[0]) + float64(ob[(2*k)*P+p])
				xx := float64(wo*o.Stride[1]-o.Padding[1]+kj*o.Dilation[1]) + float64(ob[(2*k+1)*P+p])
				m := float32(1)
				if mb != nil {
					m = mb[k*P+p]
				}
				if y <= -1 || y >= float64(H) || xx <= -1 || xx >= float64(W) {
					continue // zero
				}
				y0, x0 := int(math.Floor(y)), int(math.Floor(xx))
				ly, lx := float32(y-float64(y0)), float32(xx-float64(x0))
				hy, hx := 1-ly, 1-lx
				var w1, w2, w3, w4 float32
				var i1, i2, i3, i4 int = -1, -1, -1, -1
				if y0 >= 0 && x0 >= 0 {
					w1, i1 = hy*hx, y0*W+x0
				}
				if y0 >= 0 && x0+1 <= W-1 {
					w2, i2 = hy*lx, y0*W+x0+1
				}
				if y0+1 <= H-1 && x0 >= 0 {
					w3, i3 = ly*hx, (y0+1)*W+x0
				}
				if y0+1 <= H-1 && x0+1 <= W-1 {
					w4, i4 = ly*lx, (y0+1)*W+x0+1
				}
				for c := range C {
					plane := xb[c*H*W : (c+1)*H*W]
					var v float32
					if i1 >= 0 {
						v += w1 * plane[i1]
					}
					if i2 >= 0 {
						v += w2 * plane[i2]
					}
					if i3 >= 0 {
						v += w3 * plane[i3]
					}
					if i4 >= 0 {
						v += w4 * plane[i4]
					}
					cols[(c*K+k)*P+p] = v * m
				}
			}
		})
		ct, err := tensor.New(cols, []int{C * K, P}, dev)
		if err != nil {
			return nil, err
		}
		y, err := MatMul(wf, ct) // (O, P)
		if err != nil {
			return nil, err
		}
		if bias != nil {
			bb := bias
			if bb.DType() != dtype.F32 {
				if bb, err = Cast(bb, dtype.F32); err != nil {
					return nil, err
				}
			}
			if bb, err = bb.Reshape(O, 1); err != nil {
				return nil, err
			}
			if y, err = Add(y, bb); err != nil {
				return nil, err
			}
		}
		if outs[b], err = y.Reshape(1, O, Ho, Wo); err != nil {
			return nil, err
		}
	}
	if B == 1 {
		return outs[0], nil
	}
	return Concat(outs, 0)
}
