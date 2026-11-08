// Vendored from github.com/sushvibez/gfx (ffi package, Apache-2.0),
// without its js/wasm stub; unchanged otherwise.
//
// Package ffi is the cgo replacement: it loads native shared libraries and
// resolves symbols without a C toolchain, so the whole framework
// cross-compiles with plain `go build`.
//
// Backends (see /DESIGN.md §6):
//   - Linux / Android / macOS: ebitengine/purego (dlopen + trampolines).
//   - Windows: stdlib syscall DLL loading (no cgo needed there at all).
//   - Web (js/wasm): no native code exists; GPU and platform access go
//     through syscall/js instead, so this package is a no-op stub there.
//
// Two ways to call. Lib.Bind (purego.RegisterFunc) accepts almost any
// signature but goes through reflect.MakeFunc: ~700 ns and several
// allocations per call. Call is the fast path — up to six integer-sized
// arguments, one integer result, an assembly trampoline per ABI, ~40 ns and
// no allocation; CallStack adds up to eight words passed on the stack,
// which is how the SysV ABI takes a by-value struct larger than two words.
// cmd/ffigen generates typed stubs over Call from a
// package's Bind declarations, so hand-written code never spells Call out;
// see gpu/zffi_gen.go for what it produces.
package ffi
