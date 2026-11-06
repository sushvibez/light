package ops

import (
	"fmt"
	"math"

	"light/device"
	"light/dtype"
	"light/tensor"
)

// BatchNormOptions controls BatchNorm2d training behaviour.
//
//	Train: if true, use batch statistics and update the running mean/var;
//	       if false, use running_mean / running_var for normalisation
//	       (typical at inference).
//	Momentum: exponential-moving-average factor for running stats. PyTorch
//	       default is 0.1. Running = (1-m)*running + m*batch.
//	Eps: numerical-stability term added to variance.
type BatchNormOptions struct {
	Train    bool
	Momentum float32
	Eps      float32
}

// BatchNorm2dResult bundles the forward output with cache values the
// backward pass needs (`xHat` and `sigma` per channel).
type BatchNorm2dResult struct {
	Out         *tensor.Tensor // (N, C, H, W)
	XHat        *tensor.Tensor // (N, C, H, W) — (x - mean)/sigma
	Sigma       *tensor.Tensor // (C,)          — sqrt(var + eps) per channel
	Mean        *tensor.Tensor // (C,)          — batch mean (train) or running_mean (eval)
	RunningMean *tensor.Tensor // updated copy if Train else same as input
	RunningVar  *tensor.Tensor // updated copy if Train else same as input
}

// BatchNorm2d applies BatchNorm over the (N, H, W) axes per channel.
// `runningMean` and `runningVar` are per-channel (C,) F32 tensors. In
// training mode they are updated via EMA; in eval mode they are used as-is.
//
// CPU F32 only for the MVP.
func BatchNorm2d(
	input, gamma, beta, runningMean, runningVar *tensor.Tensor,
	opts BatchNormOptions,
) (*BatchNorm2dResult, error) {
	// Accelerator path: native kernels where registered (CUDA train-mode
	// via cnnhooks, eval-mode "batchnorm_eval" registry kernels), otherwise
	// transfer inputs to CPU, compute, and transfer outputs back.
	if input.Device().Kind() != device.KindCPU {
		return batchNorm2dDeviceFallback(input, gamma, beta, runningMean, runningVar, opts)
	}
	if input.DType() != dtype.F32 {
		return nil, fmt.Errorf("BatchNorm2d: F32 only")
	}
	if input.Rank() != 4 {
		return nil, fmt.Errorf("BatchNorm2d: input must be (N, C, H, W)")
	}
	if opts.Eps <= 0 {
		opts.Eps = 1e-5
	}
	inC, err := ensureContig(input)
	if err != nil {
		return nil, err
	}
	dims := inC.Shape().Dims()
	N, C, H, W := dims[0], dims[1], dims[2], dims[3]
	hw := H * W
	nhw := float32(N * hw)
	x := cpuF32Bytes(inC)

	// Shared buffers
	meanBuf := make([]float32, C)
	varBuf := make([]float32, C)
	sigmaBuf := make([]float32, C)

	// Compute channel-wise mean and variance from the batch when training,
	// or reuse the running stats otherwise.
	if opts.Train {
		for c := range C {
			var sum, sumSq float64
			for n := range N {
				base := (n*C + c) * hw
				for k := range hw {
					v := x[base+k]
					sum += float64(v)
					sumSq += float64(v) * float64(v)
				}
			}
			m := sum / float64(nhw)
			v := sumSq/float64(nhw) - m*m
			meanBuf[c] = float32(m)
			varBuf[c] = float32(v)
			sigmaBuf[c] = float32(math.Sqrt(v + float64(opts.Eps)))
		}
	} else {
		rm := cpuF32Bytes(runningMean)
		rv := cpuF32Bytes(runningVar)
		for c := range C {
			meanBuf[c] = rm[c]
			varBuf[c] = rv[c]
			sigmaBuf[c] = float32(math.Sqrt(float64(rv[c]) + float64(opts.Eps)))
		}
	}

	gammaV := cpuF32Bytes(gamma)
	betaV := cpuF32Bytes(beta)

	out, err := tensor.Zeros(dtype.F32, dims, inC.Device())
	if err != nil {
		return nil, err
	}
	xHat, err := tensor.Zeros(dtype.F32, dims, inC.Device())
	if err != nil {
		return nil, err
	}
	y := cpuF32Bytes(out)
	xh := cpuF32Bytes(xHat)

	for n := range N {
		for c := range C {
			mu := meanBuf[c]
			sigma := sigmaBuf[c]
			g := gammaV[c]
			b := betaV[c]
			base := (n*C + c) * hw
			for k := range hw {
				xhv := (x[base+k] - mu) / sigma
				xh[base+k] = xhv
				y[base+k] = xhv*g + b
			}
		}
	}

	sigmaT, _ := tensor.New(sigmaBuf, []int{C}, inC.Device())
	meanT, _ := tensor.New(meanBuf, []int{C}, inC.Device())

	// Running-stats update (in place on a copy).
	rmOut, err := copyTensor(runningMean)
	if err != nil {
		return nil, err
	}
	rvOut, err := copyTensor(runningVar)
	if err != nil {
		return nil, err
	}
	if opts.Train {
		rm := cpuF32Bytes(rmOut)
		rv := cpuF32Bytes(rvOut)
		mom := opts.Momentum
		if mom == 0 {
			mom = 0.1
		}
		// Unbiased variance for running stats (matches PyTorch).
		biasCorr := float32(1)
		if nhw > 1 {
			biasCorr = nhw / (nhw - 1)
		}
		for c := range C {
			rm[c] = (1-mom)*rm[c] + mom*meanBuf[c]
			rv[c] = (1-mom)*rv[c] + mom*varBuf[c]*biasCorr
		}
	}

	return &BatchNorm2dResult{
		Out:         out,
		XHat:        xHat,
		Sigma:       sigmaT,
		Mean:        meanT,
		RunningMean: rmOut,
		RunningVar:  rvOut,
	}, nil
}

// BatchNorm2dBackward returns (grad_input, grad_gamma, grad_beta).
// Uses the cached x_hat and sigma from forward — faster and more stable than
// recomputing from x, gamma, beta, mean, var. CPU F32 only.
func BatchNorm2dBackward(
	gradOut, xHat, sigma, gamma *tensor.Tensor,
) (gradIn, gradGamma, gradBeta *tensor.Tensor, err error) {
	if gradOut.Device().Kind() != device.KindCPU {
		return batchNorm2dBackwardDeviceFallback(gradOut, xHat, sigma, gamma)
	}
	gC, err := ensureContig(gradOut)
	if err != nil {
		return nil, nil, nil, err
	}
	xh, err := ensureContig(xHat)
	if err != nil {
		return nil, nil, nil, err
	}
	dims := gC.Shape().Dims()
	N, C, H, W := dims[0], dims[1], dims[2], dims[3]
	hw := H * W
	m := float32(N * hw)

	g := cpuF32Bytes(gC)
	xhv := cpuF32Bytes(xh)
	sig := cpuF32Bytes(sigma)
	gam := cpuF32Bytes(gamma)

	// grad_gamma[c] = sum(g * xHat) over (N, H, W)
	// grad_beta[c]  = sum(g)         over (N, H, W)
	gGamma, _ := tensor.Zeros(dtype.F32, []int{C}, gC.Device())
	gBeta, _ := tensor.Zeros(dtype.F32, []int{C}, gC.Device())
	ggv := cpuF32Bytes(gGamma)
	gbv := cpuF32Bytes(gBeta)
	for c := range C {
		var sGamma, sBeta float64
		for n := range N {
			base := (n*C + c) * hw
			for k := range hw {
				gv := g[base+k]
				sGamma += float64(gv) * float64(xhv[base+k])
				sBeta += float64(gv)
			}
		}
		ggv[c] = float32(sGamma)
		gbv[c] = float32(sBeta)
	}

	// grad_input derivation (training-time):
	//   dy_nk = g * γ
	//   dx_hat_nk = dy_nk
	//   dx_nk = (1/σ) * (dx_hat_nk - mean(dx_hat) - x_hat_nk * mean(dx_hat * x_hat))
	// where the means are over (N, H, W) per channel.
	gIn, _ := tensor.Zeros(dtype.F32, dims, gC.Device())
	giv := cpuF32Bytes(gIn)
	for c := range C {
		gammaC := gam[c]
		sigmaC := sig[c]
		// accumulate means.
		var sumGP, sumGPxHat float64
		for n := range N {
			base := (n*C + c) * hw
			for k := range hw {
				gp := g[base+k] * gammaC
				sumGP += float64(gp)
				sumGPxHat += float64(gp) * float64(xhv[base+k])
			}
		}
		meanGP := float32(sumGP / float64(m))
		meanGPxHat := float32(sumGPxHat / float64(m))
		for n := range N {
			base := (n*C + c) * hw
			for k := range hw {
				gp := g[base+k] * gammaC
				giv[base+k] = (gp - meanGP - xhv[base+k]*meanGPxHat) / sigmaC
			}
		}
	}

	return gIn, gGamma, gBeta, nil
}

// copyTensor makes an independent deep copy of a CPU F32 tensor. Needed so
// running-stats updates don't mutate the user-visible tensor in place.
func copyTensor(t *tensor.Tensor) (*tensor.Tensor, error) {
	if t == nil {
		return nil, nil
	}
	src, err := ensureContig(t)
	if err != nil {
		return nil, err
	}
	dst, err := tensor.Zeros(t.DType(), t.Shape().Dims(), t.Device())
	if err != nil {
		return nil, err
	}
	copy(dst.Storage().CPUBytes(), src.Storage().CPUBytes())
	return dst, nil
}
