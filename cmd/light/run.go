package main

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
	"sync"

	"light/device"
	"light/engine"
)

// modelFlags are the load options shared by run and serve.
type modelFlags struct{ device, dtype string }

func (f *modelFlags) register(fs *flag.FlagSet) {
	fs.StringVar(&f.device, "device", "", "gpu (CUDA), webgpu (any GPU through wgpu-native) or cpu (default: CUDA when present, else cpu)")
	fs.StringVar(&f.dtype, "dtype", "", "f32 or bf16 for safetensors weights; on WebGPU bf16 / f16 store the matmul weights in 16 bits under F32 compute (default: bf16 on the GPU and WebGPU, f32 on the CPU up to 6 GB)")
}

// spec is e's load spec with the flags applied. On the CPU, safetensors
// models up to 6 GB load in F32 unless -dtype says otherwise: the CPU BF16
// path is several times slower.
func (f *modelFlags) spec(e engine.Entry) engine.Spec {
	s := e.Spec
	s.Device = f.device
	if s.Device == "" && !device.CUDAAvailable() {
		s.Device = "cpu"
	}
	switch {
	case f.dtype != "":
		s.DType = f.dtype
	case s.Device == "webgpu":
		s.DType = "bf16" // F32 compute over BF16 weight storage (half the memory of f32)
	case s.Device == "cpu" && (e.Kind == "safetensors" || e.Kind == "pytorch") && e.SizeBytes <= 6<<30:
		s.DType = "f32" // the CPU BF16 path is slower than F32 for models that fit
	}
	return s
}

func cmdRun(args []string) error {
	fs := flag.NewFlagSet("run", flag.ContinueOnError)
	var mf modelFlags
	mf.register(fs)
	temp := fs.Float64("temp", 0.7, "sampling temperature (0: greedy)")
	topP := fs.Float64("top-p", 0.9, "nucleus sampling mass")
	topK := fs.Int("top-k", 40, "sample from the k most likely tokens (0: all)")
	maxTok := fs.Int("max-tokens", 1024, "most tokens per reply")
	seed := fs.Uint64("seed", 0, "sampling seed (0: random)")
	system := fs.String("system", "", "system prompt")
	verbose := fs.Bool("verbose", false, "print timing after each reply")
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: light run [flags] <model> [prompt]\n\nWith a prompt argument or piped stdin, prints one reply and exits;\notherwise starts an interactive chat.\n\nflags:")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 {
		fs.Usage()
		return errors.New("missing model")
	}
	entries, err := catalog()
	if err != nil {
		return err
	}
	e, err := resolve(entries, fs.Arg(0), chatKinds)
	if err != nil {
		return err
	}
	// The prompt is the arguments after the model, or else piped stdin
	// (stdin is not read when a prompt is given: an open pipe that never
	// closes would block).
	prompt := strings.Join(fs.Args()[1:], " ")
	if prompt == "" {
		if st, err := os.Stdin.Stat(); err == nil && st.Mode()&os.ModeCharDevice == 0 {
			in, err := io.ReadAll(os.Stdin)
			if err != nil {
				return err
			}
			prompt = strings.TrimSpace(string(in))
			if prompt == "" {
				return errors.New("empty prompt on stdin")
			}
		}
	}
	oneShot := prompt != ""

	rt := engine.NewRuntime()
	opts := engine.GenOpts{MaxTokens: *maxTok, Temperature: float32(*temp), TopP: float32(*topP), TopK: *topK, Seed: *seed}
	if oneShot {
		m, err := loadMeasured(rt, mf.spec(e))
		if err != nil {
			return err
		}
		var msgs []engine.Message
		if *system != "" {
			msgs = append(msgs, engine.Message{Role: "system", Content: *system})
		}
		c := &chat{rt: rt, opts: opts, verbose: *verbose, onStats: func(st engine.Stats) { recordReply(m, st, "run") }}
		_, err = c.reply(append(msgs, engine.Message{Role: "user", Content: prompt}))
		return err
	}
	sh := newShell(rt, mf, opts, *system, *verbose)
	if err := sh.load(e); err != nil {
		return err
	}
	fmt.Fprintln(os.Stderr, "/help for commands, Ctrl-D to quit.")
	return sh.run()
}

// defaultOpts are `light run`'s flag defaults, used by the bare shell.
var defaultOpts = engine.GenOpts{MaxTokens: 1024, Temperature: 0.7, TopP: 0.9, TopK: 40}

// cmdShell is `light` with no arguments: the interactive shell, no model
// loaded yet.
func cmdShell() error {
	sh := newShell(engine.NewRuntime(), modelFlags{}, defaultOpts, "", false)
	fmt.Fprintln(os.Stderr, "Light — /model <name> loads a chat model, /model lists them, /help for all commands, Ctrl-D to quit.")
	return sh.run()
}

// chat streams replies and lets Ctrl-C stop the current one.
type chat struct {
	rt      *engine.Runtime
	opts    engine.GenOpts
	onStats func(engine.Stats) // called after each completed reply (the perf table)
	verbose bool

	mu     sync.Mutex
	cancel context.CancelFunc
}

// reply generates the assistant's answer to msgs, streaming it to stdout.
func (c *chat) reply(msgs []engine.Message) (string, error) {
	ctx, cancel := context.WithCancel(context.Background())
	c.mu.Lock()
	c.cancel = cancel
	c.mu.Unlock()
	defer func() {
		c.mu.Lock()
		c.cancel = nil
		c.mu.Unlock()
		cancel()
	}()
	out := bufio.NewWriter(os.Stdout)
	text, st, err := c.rt.Generate(ctx, msgs, c.opts, func(delta string) {
		out.WriteString(delta)
		out.Flush()
	})
	out.WriteString("\n")
	out.Flush()
	if err != nil {
		return text, err
	}
	if c.onStats != nil && st.Stop != "cancel" {
		c.onStats(st)
	}
	if st.Stop == "cancel" {
		fmt.Fprintln(os.Stderr, "[stopped]")
	}
	if c.verbose {
		line := fmt.Sprintf("[%d prompt tokens in %.0f ms; %d tokens at %.1f tok/s; stop: %s", st.PromptTokens, st.PrefillMS, st.Tokens, st.TokPerSec, st.Stop)
		if st.Drafted > 0 {
			line += fmt.Sprintf("; %d / %d drafts accepted", st.Accepted, st.Drafted)
		}
		fmt.Fprintln(os.Stderr, line+"]")
	}
	return text, nil
}

// interrupt stops the reply in progress; it reports whether there was one.
func (c *chat) interrupt() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.cancel == nil {
		return false
	}
	c.cancel()
	return true
}

// readMessage reads one line, or a block between lines holding only `"""`.
func readMessage(in *bufio.Reader) (string, error) {
	line, err := in.ReadString('\n')
	if err != nil && (err != io.EOF || line == "") {
		return "", err
	}
	line = strings.TrimRight(line, "\r\n")
	if strings.TrimSpace(line) != `"""` {
		return line, nil
	}
	var b strings.Builder
	for {
		fmt.Fprint(os.Stderr, "... ")
		l, err := in.ReadString('\n')
		l = strings.TrimRight(l, "\r\n")
		if strings.TrimSpace(l) == `"""` {
			return b.String(), nil
		}
		if b.Len() > 0 {
			b.WriteString("\n")
		}
		b.WriteString(l)
		if err != nil {
			return b.String(), nil
		}
	}
}
