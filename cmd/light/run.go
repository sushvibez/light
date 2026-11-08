package main

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"sync"

	"light/device"
	"light/engine"
)

// modelFlags are the load options shared by run and serve.
type modelFlags struct{ device, dtype string }

func (f *modelFlags) register(fs *flag.FlagSet) {
	fs.StringVar(&f.device, "device", "", "gpu or cpu (default: the GPU when this build has CUDA and one is present)")
	fs.StringVar(&f.dtype, "dtype", "", "f32 or bf16 for safetensors weights (default: bf16 on the GPU, f32 on the CPU up to 6 GB)")
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
	case s.Device == "cpu" && (e.Kind == "safetensors" || e.Kind == "pytorch") && e.SizeBytes <= 6<<30:
		s.DType = "f32"
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
	if !oneShot {
		fmt.Fprintf(os.Stderr, "loading %s …\n", e.Spec.ID)
	}
	m, err := rt.Load(mf.spec(e))
	if err != nil {
		return err
	}
	opts := engine.GenOpts{MaxTokens: *maxTok, Temperature: float32(*temp), TopP: float32(*topP), TopK: *topK, Seed: *seed}
	var msgs []engine.Message
	if *system != "" {
		msgs = append(msgs, engine.Message{Role: "system", Content: *system})
	}
	c := &chat{rt: rt, opts: opts, verbose: *verbose}
	if oneShot {
		_, err := c.reply(append(msgs, engine.Message{Role: "user", Content: prompt}))
		return err
	}
	fmt.Fprintf(os.Stderr, "%s ready (%s template, %.1f s). /help for commands, Ctrl-D to quit.\n",
		m.Spec.ID, m.Template.Source, m.LoadTime.Seconds())
	return c.loop(msgs, *system)
}

// chat streams replies and lets Ctrl-C stop the current one.
type chat struct {
	rt      *engine.Runtime
	opts    engine.GenOpts
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

const chatHelp = `commands:
  /clear           forget the conversation
  /system <text>   set the system prompt (and clear)
  /verbose         toggle timing after each reply
  /bye             quit (or Ctrl-D)
  """              start / end a multi-line message
Ctrl-C stops a reply in progress.`

func (c *chat) loop(msgs []engine.Message, system string) error {
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt)
	defer signal.Stop(sig)
	go func() {
		for range sig {
			if !c.interrupt() {
				fmt.Fprint(os.Stderr, "\n(/bye or Ctrl-D to quit)\n>>> ")
			}
		}
	}()
	in := bufio.NewReader(os.Stdin)
	for {
		fmt.Fprint(os.Stderr, ">>> ")
		line, err := readMessage(in)
		if err == io.EOF {
			fmt.Fprintln(os.Stderr)
			return nil
		}
		if err != nil {
			return err
		}
		switch cmd := strings.TrimSpace(line); {
		case cmd == "":
			continue
		case cmd == "/bye" || cmd == "/exit" || cmd == "/quit":
			return nil
		case cmd == "/help" || cmd == "/?":
			fmt.Fprintln(os.Stderr, chatHelp)
			continue
		case cmd == "/clear":
			msgs = msgs[:0]
			if system != "" {
				msgs = append(msgs, engine.Message{Role: "system", Content: system})
			}
			fmt.Fprintln(os.Stderr, "(cleared)")
			continue
		case cmd == "/verbose":
			c.verbose = !c.verbose
			fmt.Fprintf(os.Stderr, "(verbose %v)\n", c.verbose)
			continue
		case strings.HasPrefix(cmd, "/system"):
			system = strings.TrimSpace(strings.TrimPrefix(cmd, "/system"))
			msgs = msgs[:0]
			if system != "" {
				msgs = append(msgs, engine.Message{Role: "system", Content: system})
			}
			fmt.Fprintln(os.Stderr, "(system prompt set; conversation cleared)")
			continue
		case strings.HasPrefix(cmd, "/"):
			fmt.Fprintf(os.Stderr, "unknown command %s; /help lists them\n", cmd)
			continue
		}
		msgs = append(msgs, engine.Message{Role: "user", Content: line})
		text, err := c.reply(msgs)
		if err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
			msgs = msgs[:len(msgs)-1]
			continue
		}
		msgs = append(msgs, engine.Message{Role: "assistant", Content: text})
	}
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
