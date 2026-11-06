package ops

import (
	"fmt"

	"light/device"
	"light/internal/cnnhooks"
	"light/internal/registry"
	"light/tensor"
)

// batchNorm2dDeviceFallback runs BatchNorm2d on a non-CPU device. When the
// native CUDA train-mode kernels are registered, the forward uses them and
// running-stats are updated in Go via existing ops (one elementwise pass).
// Eval mode uses a registered "batchnorm_eval" kernel when the device has
// one (WebGPU does); otherwise — and for the CPU build — the reference CPU
// path runs via a single stage-on-CPU / compute / ship-back round trip.
func batchNorm2dDeviceFallback(
	input, gamma, beta, runningMean, runningVar *tensor.Tensor,
	opts BatchNormOptions,
) (*BatchNorm2dResult, error) {
	if opts.Train && input.Device().Kind() == device.KindCUDA && cnnhooks.BatchNorm2dForwardCUDA != nil {
		return batchNorm2dCUDANative(input, gamma, beta, runningMean, runningVar, opts)
	}
	if !opts.Train {
		if k, err := registry.Lookup(registry.Key{Op: "batchnorm_eval", Dev: input.Device().Kind(), DType: input.DType()}); err == nil {
			return batchNorm2dEvalKernel(k, input, gamma, beta, runningMean, runningVar, opts)
		}
	}
	cpu := device.CPU()
	gpu := input.Device()

	inCPU, err := input.To(cpu)
	if err != nil {
		return nil, err
	}
	gammaCPU, err := gamma.To(cpu)
	if err != nil {
		return nil, err
	}
	betaCPU, err := beta.To(cpu)
	if err != nil {
		return nil, err
	}
	rmCPU, err := runningMean.To(cpu)
	if err != nil {
		return nil, err
	}
	rvCPU, err := runningVar.To(cpu)
	if err != nil {
		return nil, err
	}

	// Recurse into the CPU path by re-invoking BatchNorm2d — with CPU
	// inputs it takes the non-fallback branch.
	resCPU, err := BatchNorm2d(inCPU, gammaCPU, betaCPU, rmCPU, rvCPU, opts)
	if err != nil {
		return nil, err
	}

	outGPU, err := resCPU.Out.To(gpu)
	if err != nil {
		return nil, err
	}
	xHatGPU, err := resCPU.XHat.To(gpu)
	if err != nil {
		return nil, err
	}
	sigmaGPU, err := resCPU.Sigma.To(gpu)
	if err != nil {
		return nil, err
	}
	meanGPU, err := resCPU.Mean.To(gpu)
	if err != nil {
		return nil, err
	}
	rmOut, err := resCPU.RunningMean.To(gpu)
	if err != nil {
		return nil, err
	}
	rvOut, err := resCPU.RunningVar.To(gpu)
	if err != nil {
		return nil, err
	}
	return &BatchNorm2dResult{
		Out:         outGPU,
		XHat:        xHatGPU,
		Sigma:       sigmaGPU,
		Mean:        meanGPU,
		RunningMean: rmOut,
		RunningVar:  rvOut,
	}, nil
}

// batchNorm2dCUDANative uses the train-mode CUDA kernels, with Go-side
// running-stats update via elementwise ops.
func batchNorm2dCUDANative(
	input, gamma, beta, runningMean, runningVar *tensor.Tensor,
	opts BatchNormOptions,
) (*BatchNorm2dResult, error) {
	if opts.Eps == 0 {
		opts.Eps = 1e-5
	}
	fwd, err := cnnhooks.BatchNorm2dForwardCUDA(input, gamma, beta, opts.Eps)
	if err != nil {
		return nil, err
	}
	// Running stats: rm_new = (1-m)*rm + m*mean; rv_new = (1-m)*rv + m*var*biasCorr
	mom := opts.Momentum
	if mom == 0 {
		mom = 0.1
	}
	dims := input.Shape().Dims()
	nhw := float32(dims[0] * dims[2] * dims[3])
	biasCorr := float32(1)
	if nhw > 1 {
		biasCorr = nhw / (nhw - 1)
	}
	oneMinusMom, err := scalarFull(1-mom, input.Device())
	if err != nil {
		return nil, err
	}
	momT, err := scalarFull(mom, input.Device())
	if err != nil {
		return nil, err
	}
	correctedMomT, err := scalarFull(mom*biasCorr, input.Device())
	if err != nil {
		return nil, err
	}
	// rm_new = oneMinusMom * rm + mom * mean
	rmPart1, err := Mul(runningMean, oneMinusMom)
	if err != nil {
		return nil, err
	}
	rmPart2, err := Mul(fwd.Mean, momT)
	if err != nil {
		return nil, err
	}
	rmNew, err := Add(rmPart1, rmPart2)
	if err != nil {
		return nil, err
	}
	// rv_new = oneMinusMom * rv + mom*biasCorr * var
	rvPart1, err := Mul(runningVar, oneMinusMom)
	if err != nil {
		return nil, err
	}
	rvPart2, err := Mul(fwd.Var, correctedMomT)
	if err != nil {
		return nil, err
	}
	rvNew, err := Add(rvPart1, rvPart2)
	if err != nil {
		return nil, err
	}
	return &BatchNorm2dResult{
		Out:         fwd.Out,
		XHat:        fwd.XHat,
		Sigma:       fwd.Sigma,
		Mean:        fwd.Mean,
		RunningMean: rmNew,
		RunningVar:  rvNew,
	}, nil
}

// scalarFull builds a [1] F32 tensor on the given device.
func scalarFull(v float32, dev device.Device) (*tensor.Tensor, error) {
	return tensor.Full(v, []int{1}, dev)
}

// batchNorm2dEvalKernel runs an eval-mode BatchNorm2d through a registered
// per-device kernel (inputs x, gamma, beta, running_mean, running_var; attrs
// eps and xhat). Two launches produce Out and XHat; Sigma is derived with
// elementwise ops so everything stays on device.
func batchNorm2dEvalKernel(
	k registry.Kernel,
	input, gamma, beta, runningMean, runningVar *tensor.Tensor,
	opts BatchNormOptions,
) (*BatchNorm2dResult, error) {
	if input.Rank() != 4 {
		return nil, fmt.Errorf("BatchNorm2d: input must be (N, C, H, W)")
	}
	if opts.Eps <= 0 {
		opts.Eps = 1e-5
	}
	ins := make([]*tensor.Tensor, 0, 5)
	for _, t := range []*tensor.Tensor{input, gamma, beta, runningMean, runningVar} {
		c, err := ensureContig(t)
		if err != nil {
			return nil, err
		}
		ins = append(ins, c)
	}
	dims := ins[0].Shape().Dims()
	out, err := tensor.Zeros(input.DType(), dims, input.Device())
	if err != nil {
		return nil, err
	}
	if err := k(&registry.Context{Inputs: ins, Output: out, Attrs: map[string]any{"eps": float64(opts.Eps), "xhat": false}}); err != nil {
		return nil, err
	}
	xHat, err := tensor.Zeros(input.DType(), dims, input.Device())
	if err != nil {
		return nil, err
	}
	if err := k(&registry.Context{Inputs: ins, Output: xHat, Attrs: map[string]any{"eps": float64(opts.Eps), "xhat": true}}); err != nil {
		return nil, err
	}
	epsT, err := scalarFull(opts.Eps, input.Device())
	if err != nil {
		return nil, err
	}
	varEps, err := Add(ins[4], epsT)
	if err != nil {
		return nil, err
	}
	sigma, err := Sqrt(varEps)
	if err != nil {
		return nil, err
	}
	return &BatchNorm2dResult{
		Out:         out,
		XHat:        xHat,
		Sigma:       sigma,
		Mean:        ins[3],
		RunningMean: runningMean,
		RunningVar:  runningVar,
	}, nil
}

// batchNorm2dBackwardDeviceFallback mirrors the forward fallback.
func batchNorm2dBackwardDeviceFallback(
	gradOut, xHat, sigma, gamma *tensor.Tensor,
) (*tensor.Tensor, *tensor.Tensor, *tensor.Tensor, error) {
	if gradOut.Device().Kind() == device.KindCUDA && cnnhooks.BatchNorm2dBackwardCUDA != nil {
		return cnnhooks.BatchNorm2dBackwardCUDA(gradOut, xHat, sigma, gamma)
	}
	cpu := device.CPU()
	gpu := gradOut.Device()

	gCPU, err := gradOut.To(cpu)
	if err != nil {
		return nil, nil, nil, err
	}
	xhCPU, err := xHat.To(cpu)
	if err != nil {
		return nil, nil, nil, err
	}
	sigmaCPU, err := sigma.To(cpu)
	if err != nil {
		return nil, nil, nil, err
	}
	gammaCPU, err := gamma.To(cpu)
	if err != nil {
		return nil, nil, nil, err
	}

	gIn, gGamma, gBeta, err := BatchNorm2dBackward(gCPU, xhCPU, sigmaCPU, gammaCPU)
	if err != nil {
		return nil, nil, nil, err
	}
	gInGPU, err := gIn.To(gpu)
	if err != nil {
		return nil, nil, nil, err
	}
	gGammaGPU, err := gGamma.To(gpu)
	if err != nil {
		return nil, nil, nil, err
	}
	gBetaGPU, err := gBeta.To(gpu)
	if err != nil {
		return nil, nil, nil, err
	}
	return gInGPU, gGammaGPU, gBetaGPU, nil
}
