# XGo experiments

This directory contains a few small training / inference scripts written in
[**XGo**](https://github.com/goplus/xgo) (formerly *Go+*) instead of plain Go.
The point is to compare ergonomics: XGo has terser syntax and Python-style
idioms while still being a Go-compatible language that compiles to the same
runtime — including our CUDA backend.

**Crucially, none of the `light/...` packages were modified.** The XGo files
just `import` them and call their public APIs the same way Go code does.
XGo is a strict superset, so the transpiled `.go` files end up using the
exact same `nn.Linear`, `autograd.Backward`, etc. that `examples/11_train_mlp`
uses. Performance is identical.

## Files

| File | Description |
|------|-------------|
| `01_train_mlp/main.xgo` | 4-class MLP on synthetic data; 200 AdamW steps; loss drops 1.38 → 0.01. |
| `02_infer_gpt2/main.xgo` | Loads `gpt2` from HuggingFace, greedy-decodes a prompt. Mirrors `examples/25_gpt2_pretrained`. |

## Run them

```bash
xgo run ./examples/xgo/01_train_mlp
xgo run ./examples/xgo/02_infer_gpt2
```

`xgo run` transpiles to Go on the fly, then `go run`s the result. Cached
generated `.go` files end up next to the `.xgo` source — they're already
gitignored.

## What XGo gives you

Side-by-side, the same training loop:

```go
// Plain Go (examples/11_train_mlp/main.go):
for step := 0; step < 200; step++ {
    opt.ZeroGrad()
    logits, err := model.Forward(x)
    if err != nil { log.Fatal(err) }
    loss, err := autograd.CrossEntropyLoss(logits, y)
    if err != nil { log.Fatal(err) }
    if err := autograd.Backward(loss, nil); err != nil { log.Fatal(err) }
    if err := opt.Step(); err != nil { log.Fatal(err) }
    if step%20 == 0 {
        lc, _ := loss.To(device.CPU())
        fmt.Printf("step %d  loss=%.4f\n", step, tensor.MustSlice[float32](lc)[0])
    }
}
```

```xgo
// XGo (examples/xgo/01_train_mlp.xgo):
for step <- 0:200 {
    opt.ZeroGrad()
    logits := forward(x)
    loss, _ := autograd.CrossEntropyLoss(logits, y)
    autograd.Backward(loss, nil)
    opt.Step()
    if step % 20 == 0 {
        lc, _ := loss.To(device.CPU())
        println "step", step, "loss=", tensor.MustSlice[float32](lc)[0]
    }
}
```

The XGo version is shorter because:
- `for step <- 0:200` replaces `for step := 0; step < 200; step++`.
- `println` formats arguments without explicit `fmt.Printf` placeholders.
- Single-return calls don't need parens or `var := ...`.
- No `if err != nil` boilerplate when you're prototyping; XGo lets you
  silently drop errors with `_` and panic-on-error semantics in scripts.

## When (not) to use XGo

XGo works well for **scripts** — quick experiments, one-off training runs,
notebook-style exploration. You get the speed and type safety of Go with
syntax that reads more like a Python script.

For **library** code, plain Go is still preferable: the explicit error
handling, the larger third-party ecosystem (linters, IDE support, profilers),
and a no-surprises mental model are worth more than a few saved keystrokes.

The pragmatic recommendation that motivated this directory: keep the core
of the project (under `nn/`, `autograd/`, `kernels/`, …) in plain Go, and use
XGo for the user-facing training/inference scripts where readability wins.
