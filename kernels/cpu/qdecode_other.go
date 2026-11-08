//go:build !amd64

package cpu

const decodeAVX2 = false

func q8_0RowAVX2(raw *byte, d *float32, dst *float32, nb int)       { panic("unreachable") }
func q4_0RowAVX2(raw *byte, d *float32, dst *float32, nb int)       { panic("unreachable") }
func q5_0RowAVX2(raw *byte, d *float32, dst *float32, nb int)       { panic("unreachable") }
func q4KBlockAVX2(qs *byte, sc *float32, mn *float32, dst *float32) { panic("unreachable") }
func q6KBlockAVX2(ql *byte, qh *byte, sc *float32, dst *float32)    { panic("unreachable") }
func int4RowAVX2(codes *uint32, s *float32, z *float32, dst *float32, groups, wpg int) {
	panic("unreachable")
}
