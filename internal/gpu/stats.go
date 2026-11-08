package gpu

import (
	"fmt"
	"os"
	"sync/atomic"
	"time"
)

// Upload counters: how much a frame hands to the driver, and in how many
// calls. Every call into wgpu crosses the Go/C boundary (ffi.Call, ~55 ns
// through the generated stubs), and an
// upload also costs a staging copy, so both numbers matter when a frame
// looks expensive for the little it draws. Free to keep on: two atomic
// adds per call.
//
// Read them with UploadStats and clear them per frame with ResetUploadStats
// (the shell does, under GFX_FPS).
var (
	bufCalls, bufBytes atomic.Int64
	texCalls, texBytes atomic.Int64
	bufNanos, texNanos atomic.Int64
	// Encoding calls per frame: every one crosses into wgpu.
	callDraw, callBind, callPipe, callScissor, callVBuf, callEncoders, callPasses atomic.Int64
)

// CallStats reports the render-encoding calls since the last reset:
// draws, bind-group sets, pipeline sets, scissor sets, vertex-buffer sets,
// command encoders created, render passes begun.
func CallStats() (draw, bind, pipe, scissor, vbuf, encoders, passes int64) {
	return callDraw.Load(), callBind.Load(), callPipe.Load(), callScissor.Load(),
		callVBuf.Load(), callEncoders.Load(), callPasses.Load()
}

// callTrace prints the per-frame call counts once a second from Submit
// (GFX_CALLS=1), for loops that do not go through the ui shell's GFX_FPS
// line (the games' own frame loops).
var callTrace = os.Getenv("GFX_CALLS") != ""

var callTraceAt time.Time
var callTraceFrames int

func noteSubmitTrace() {
	if !callTrace {
		return
	}
	callTraceFrames++
	now := time.Now()
	if callTraceAt.IsZero() {
		// The first submit starts the window: start-up uploads are left
		// out, and so is the submit itself, whose counts were just reset.
		callTraceAt, callTraceFrames = now, 0
		ResetUploadStats()
		return
	}
	if now.Sub(callTraceAt) < time.Second {
		return
	}
	reportCalls(now)
}

// reportCalls prints the averages since callTraceAt and starts a new window.
func reportCalls(now time.Time) {
	// Averages to a tenth: one draw a frame is 1.0, not the 0 an integer
	// division made of 60 draws over 61 submits.
	n := float64(max(callTraceFrames, 1))
	per := func(v int64) float64 { return float64(v) / n }
	dr, bi, pi, sc, vb, en, pa := CallStats()
	bc, bb, tc, tb := UploadStats()
	fmt.Fprintf(os.Stderr, "gpu: per submit: %.1f draws, %.1f bind groups, %.1f pipelines, %.1f scissors, %.1f vertex bufs, %.1f encoders, %.1f passes | %.1f buffer uploads (%.0f KB), %.1f texture uploads (%.0f KB) | %d submits in %v\n",
		per(dr), per(bi), per(pi), per(sc), per(vb), per(en), per(pa), per(bc), per(bb)/1024, per(tc), per(tb)/1024,
		callTraceFrames, now.Sub(callTraceAt).Round(time.Millisecond))
	ResetUploadStats()
	callTraceAt, callTraceFrames = now, 0
}

// FlushCallTrace prints what GFX_CALLS has counted since its last line, so
// a run shorter than the one-second window (a -screenshot, a test) still
// reports. Device.Release calls it; a program that exits without releasing
// its device can call it itself. A no-op without GFX_CALLS.
func FlushCallTrace() {
	if !callTrace || callTraceFrames == 0 {
		return
	}
	reportCalls(time.Now())
}

// statTiming also times the uploads (GFX_UPLOAD_TIMING=1). Off by
// default: two clock reads per call are not free.
var statTiming = os.Getenv("GFX_UPLOAD_TIMING") != ""

// UploadStats reports (buffer calls, buffer bytes, texture calls, texture
// bytes) since the last ResetUploadStats.
func UploadStats() (bc, bb, tc, tb int64) {
	return bufCalls.Load(), bufBytes.Load(), texCalls.Load(), texBytes.Load()
}

// UploadNanos reports how long those calls took, when GFX_UPLOAD_TIMING is
// set (0 otherwise).
func UploadNanos() (buf, tex int64) { return bufNanos.Load(), texNanos.Load() }

// ResetUploadStats zeroes the counters.
func ResetUploadStats() {
	bufCalls.Store(0)
	bufBytes.Store(0)
	texCalls.Store(0)
	texBytes.Store(0)
	bufNanos.Store(0)
	texNanos.Store(0)
	for _, c := range []*atomic.Int64{&callDraw, &callBind, &callPipe, &callScissor, &callVBuf, &callEncoders, &callPasses} {
		c.Store(0)
	}
}

func noteBufferUpload(n int) {
	bufCalls.Add(1)
	bufBytes.Add(int64(n))
}

func noteTextureUpload(n int) {
	texCalls.Add(1)
	texBytes.Add(int64(n))
}

// uploadTimer returns a func to call when the upload is done.
func uploadTimer(tex bool) func() {
	if !statTiming {
		return func() {}
	}
	t0 := time.Now()
	return func() {
		d := int64(time.Since(t0))
		if tex {
			texNanos.Add(d)
			return
		}
		bufNanos.Add(d)
	}
}
