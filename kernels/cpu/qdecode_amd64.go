//go:build amd64

package cpu

import "light/internal/simd"

//go:noescape
func q8_0RowAVX2(raw *byte, d *float32, dst *float32, nb int)

//go:noescape
func q4_0RowAVX2(raw *byte, d *float32, dst *float32, nb int)

//go:noescape
func q5_0RowAVX2(raw *byte, d *float32, dst *float32, nb int)

//go:noescape
func q4KBlockAVX2(qs *byte, sc *float32, mn *float32, dst *float32)

//go:noescape
func q6KBlockAVX2(ql *byte, qh *byte, sc *float32, dst *float32)

//go:noescape
func int4RowAVX2(codes *uint32, s *float32, z *float32, dst *float32, groups, wpg int)

// The AVX2 decoders use AVX2 integer ops and the same CPU check as simd.
var decodeAVX2 = simd.Accelerated()
