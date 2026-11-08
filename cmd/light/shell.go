package main

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"time"

	"light/engine"
)

// shell is the interactive session `light` (no arguments) and
// `light run <model>` open: plain lines chat with the loaded model, lines
// starting with / are commands.
type shell struct {
	*chat
	in     *bufio.Reader
	mf     modelFlags
	model  *engine.Entry // the loaded chat model, nil before /model
	system string
	msgs   []engine.Message
	srv    *http.Server
	srvURL string
}

type shellCmd struct {
	name, args, help string
	run              func(s *shell, args []string) error
}

var shellCmds []shellCmd

func init() {
	shellCmds = []shellCmd{
		{"model", "[name]", "load a chat model (no name: list them)", (*shell).cmdModel},
		{"list", "", "models on disk", func(s *shell, a []string) error { return cmdList(a) }},
		{"pull", "<org/repo> [file]", "download from the Hugging Face Hub", func(s *shell, a []string) error { return cmdPull(a) }},
		{"info", "[name]", "a model's architecture (default: the loaded one)", (*shell).cmdInfo},
		{"rm", "<name>", "delete a model from disk", (*shell).cmdRm},
		{"clear", "", "forget the conversation", (*shell).cmdClear},
		{"system", "[text]", "set the system prompt (and clear); no text shows it", (*shell).cmdSystem},
		{"set", "[key value]", "settings: temp, top-p, top-k, max-tokens, seed, device (gpu|webgpu|cpu), dtype", (*shell).cmdSet},
		{"verbose", "", "toggle timing after each reply", (*shell).cmdVerbose},
		{"perf", "[model]", "this host's measured RAM / GPU memory / prefill / decode per model and device", func(s *shell, a []string) error {
			return printPerf(os.Stderr, strings.Join(a, " "))
		}},
		{"bench", "[model] [devices]", "measure a model on cpu, webgpu and gpu (default: the loaded one, every device)", (*shell).cmdBench},
		{"serve", "[addr | stop]", "serve the OpenAI API in the background (default 127.0.0.1:8129)", (*shell).cmdServe},
		{"unload", "", "free the loaded model", (*shell).cmdUnload},
		{"help", "", "this list", (*shell).cmdHelp},
		{"exit", "", "quit (also /quit, /bye, Ctrl-D)", nil},
	}
}

func newShell(rt *engine.Runtime, mf modelFlags, opts engine.GenOpts, system string, verbose bool) *shell {
	s := &shell{chat: &chat{rt: rt, opts: opts, verbose: verbose}, in: bufio.NewReader(os.Stdin), mf: mf, system: system}
	s.cmdClear(nil)
	return s
}

func (s *shell) cmdHelp([]string) error {
	fmt.Fprintln(os.Stderr, "Type a message to chat with the loaded model. Commands:")
	for _, c := range shellCmds {
		fmt.Fprintf(os.Stderr, "  /%-8s %-18s %s\n", c.name, c.args, c.help)
	}
	fmt.Fprintln(os.Stderr, `  """ starts and ends a multi-line message; Ctrl-C stops a reply.`)
	return nil
}

// run reads lines until /exit or end of input.
func (s *shell) run() error {
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, os.Interrupt)
	defer signal.Stop(sig)
	go func() {
		for range sig {
			if !s.interrupt() {
				fmt.Fprint(os.Stderr, "\n(/exit or Ctrl-D to quit)\n"+s.prompt())
			}
		}
	}()
	defer s.stopServer()
	for {
		fmt.Fprint(os.Stderr, s.prompt())
		line, err := readMessage(s.in)
		if err == io.EOF {
			fmt.Fprintln(os.Stderr)
			return nil
		}
		if err != nil {
			return err
		}
		text := strings.TrimSpace(line)
		switch {
		case text == "":
			continue
		case strings.HasPrefix(text, "/"):
			if s.command(text) {
				return nil
			}
		default:
			s.say(line)
		}
	}
}

// prompt shows the loaded model's short name.
func (s *shell) prompt() string {
	if s.model == nil {
		return "> "
	}
	id := s.model.Spec.ID
	if i := strings.LastIndex(id, "/"); i >= 0 {
		id = id[i+1:]
	}
	return strings.TrimSuffix(id, ".gguf") + " > "
}

// command runs one /command line; it reports whether to quit.
func (s *shell) command(line string) bool {
	f := strings.Fields(strings.TrimPrefix(line, "/"))
	name, args := f[0], f[1:]
	switch name {
	case "exit", "quit", "bye", "q":
		return true
	case "?":
		name = "help"
	}
	var match []shellCmd
	for _, c := range shellCmds {
		if c.name == name {
			match = []shellCmd{c}
			break
		}
		if strings.HasPrefix(c.name, name) && c.run != nil {
			match = append(match, c)
		}
	}
	switch len(match) {
	case 0:
		fmt.Fprintf(os.Stderr, "unknown command /%s; /help lists them\n", name)
	case 1:
		if match[0].run == nil {
			return true
		}
		if err := match[0].run(s, args); err != nil {
			fmt.Fprintln(os.Stderr, "error:", err)
		}
	default:
		names := make([]string, len(match))
		for i, c := range match {
			names[i] = "/" + c.name
		}
		fmt.Fprintf(os.Stderr, "/%s is ambiguous: %s\n", name, strings.Join(names, ", "))
	}
	return false
}

// say sends a user message and appends the reply to the conversation.
func (s *shell) say(line string) {
	if s.model == nil {
		fmt.Fprintln(os.Stderr, "no model loaded: /model <name> loads one (/model lists them)")
		return
	}
	s.msgs = append(s.msgs, engine.Message{Role: "user", Content: line})
	text, err := s.reply(s.msgs)
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		s.msgs = s.msgs[:len(s.msgs)-1]
		return
	}
	s.msgs = append(s.msgs, engine.Message{Role: "assistant", Content: text})
}

func (s *shell) cmdModel(args []string) error {
	entries, err := catalog()
	if err != nil {
		return err
	}
	if len(args) == 0 {
		n := 0
		for _, e := range entries {
			if !chatKinds[e.Kind] {
				continue
			}
			mark := " "
			if s.model != nil && s.model.Spec.ID == e.Spec.ID {
				mark = "*"
			}
			fmt.Fprintf(os.Stderr, " %s %-58s %6s %9s\n", mark, e.Spec.ID, e.Params, gb(e.SizeBytes))
			n++
		}
		if n == 0 {
			fmt.Fprintln(os.Stderr, "no chat models on disk; /pull <org/repo> downloads one")
		}
		return nil
	}
	e, err := resolve(entries, strings.Join(args, " "), chatKinds)
	if err != nil {
		return err
	}
	return s.load(e)
}

func (s *shell) load(e engine.Entry) error {
	spec := s.mf.spec(e)
	fmt.Fprintf(os.Stderr, "loading %s (%s, %s) …\n", e.Spec.ID, orAuto(spec.Device), orAuto(spec.DType))
	m, err := loadMeasured(s.rt, spec)
	if err != nil {
		return err
	}
	s.model = &e
	s.onStats = func(st engine.Stats) { recordReply(m, st, "run") }
	s.cmdClear(nil)
	fmt.Fprintf(os.Stderr, "%s ready (%s template, %.1f s)\n", m.Spec.ID, m.Template.Source, m.LoadTime.Seconds())
	return nil
}

func orAuto(v string) string {
	if v == "" {
		return "auto"
	}
	return v
}

func (s *shell) cmdInfo(args []string) error {
	if len(args) == 0 {
		if s.model == nil {
			return errors.New("no model loaded; /info <name>")
		}
		args = []string{s.model.Spec.ID}
	}
	return cmdInfo(args)
}

func (s *shell) cmdRm(args []string) error {
	if len(args) != 1 {
		return errors.New("usage: /rm <name>")
	}
	entries, err := catalog()
	if err != nil {
		return err
	}
	e, err := resolve(entries, args[0], nil)
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "delete %s (%s)? [y/N] ", e.Spec.ID, gb(e.SizeBytes))
	ans, _ := s.in.ReadString('\n')
	if a := strings.ToLower(strings.TrimSpace(ans)); a != "y" && a != "yes" {
		fmt.Fprintln(os.Stderr, "kept")
		return nil
	}
	if s.model != nil && s.model.Spec.ID == e.Spec.ID {
		s.cmdUnload(nil)
	}
	return cmdRm([]string{"-y", e.Spec.ID})
}

func (s *shell) cmdClear([]string) error {
	s.msgs = s.msgs[:0]
	if s.system != "" {
		s.msgs = append(s.msgs, engine.Message{Role: "system", Content: s.system})
	}
	return nil
}

func (s *shell) cmdSystem(args []string) error {
	if len(args) == 0 {
		if s.system == "" {
			fmt.Fprintln(os.Stderr, "(no system prompt)")
		} else {
			fmt.Fprintln(os.Stderr, s.system)
		}
		return nil
	}
	s.system = strings.Join(args, " ")
	s.cmdClear(nil)
	fmt.Fprintln(os.Stderr, "(system prompt set; conversation cleared)")
	return nil
}

func (s *shell) cmdVerbose([]string) error {
	s.verbose = !s.verbose
	fmt.Fprintf(os.Stderr, "(verbose %v)\n", s.verbose)
	return nil
}

func (s *shell) cmdSet(args []string) error {
	o := &s.opts
	if len(args) == 0 {
		fmt.Fprintf(os.Stderr, "temp %g  top-p %g  top-k %d  max-tokens %d  seed %d  device %s  dtype %s\n",
			o.Temperature, o.TopP, o.TopK, o.MaxTokens, o.Seed, orAuto(s.mf.device), orAuto(s.mf.dtype))
		return nil
	}
	if len(args) != 2 {
		return errors.New("usage: /set <key> <value>")
	}
	k, v := args[0], args[1]
	f, ferr := strconv.ParseFloat(v, 64)
	num := func() error {
		if ferr != nil {
			return fmt.Errorf("%s wants a number, got %q", k, v)
		}
		return nil
	}
	switch k {
	case "temp", "temperature":
		if err := num(); err != nil {
			return err
		}
		o.Temperature = float32(f)
	case "top-p", "top_p":
		if err := num(); err != nil {
			return err
		}
		o.TopP = float32(f)
	case "top-k", "top_k":
		if err := num(); err != nil {
			return err
		}
		o.TopK = int(f)
	case "max-tokens", "max_tokens":
		if err := num(); err != nil {
			return err
		}
		o.MaxTokens = int(f)
	case "seed":
		n, err := strconv.ParseUint(v, 10, 64)
		if err != nil {
			return fmt.Errorf("seed wants a whole number, got %q", v)
		}
		o.Seed = n
	case "device":
		if v != "gpu" && v != "webgpu" && v != "cpu" && v != "auto" {
			return errors.New("device: gpu (CUDA), webgpu, cpu or auto")
		}
		s.mf.device = strings.TrimPrefix(v, "auto")
		fmt.Fprintln(os.Stderr, "(applies to the next /model)")
	case "dtype":
		if v != "f32" && v != "bf16" && v != "auto" {
			return errors.New("dtype: f32, bf16 or auto")
		}
		s.mf.dtype = strings.TrimPrefix(v, "auto")
		fmt.Fprintln(os.Stderr, "(applies to the next /model)")
	default:
		return fmt.Errorf("unknown setting %q (temp, top-p, top-k, max-tokens, seed, device, dtype)", k)
	}
	return nil
}

func (s *shell) cmdServe(args []string) error {
	if len(args) == 1 && args[0] == "stop" {
		if s.srv == nil {
			return errors.New("not serving")
		}
		s.stopServer()
		fmt.Fprintln(os.Stderr, "(server stopped)")
		return nil
	}
	if s.srv != nil {
		fmt.Fprintf(os.Stderr, "serving on %s (/serve stop to end)\n", s.srvURL)
		return nil
	}
	addr := "127.0.0.1:8129"
	if len(args) == 1 {
		addr = args[0]
	}
	srv := &http.Server{Addr: addr, Handler: engine.OpenAIHandler(s.rt)}
	errc := make(chan error, 1)
	go func() { errc <- srv.ListenAndServe() }()
	select {
	case err := <-errc: // bind failures come back at once
		return err
	case <-time.After(200 * time.Millisecond):
	}
	s.srv, s.srvURL = srv, "http://"+addr+"/v1"
	fmt.Fprintf(os.Stderr, "serving the OpenAI API on %s (shares the loaded model; /serve stop to end)\n", s.srvURL)
	return nil
}

func (s *shell) stopServer() {
	if s.srv == nil {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	_ = s.srv.Shutdown(ctx)
	s.srv = nil
}

func (s *shell) cmdUnload([]string) error {
	if s.model == nil {
		return errors.New("no model loaded")
	}
	if err := s.rt.Unload(); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "(unloaded %s)\n", s.model.Spec.ID)
	s.model = nil
	return nil
}

// cmdBench measures a model (default: the loaded one) on the host's
// devices. The loaded model is released first and loaded again after, so
// the benchmark has the memory to itself.
func (s *shell) cmdBench(args []string) error {
	var devs []string
	var name []string
	for _, a := range args {
		if strings.Contains(a, ",") || a == "cpu" || a == "webgpu" || a == "gpu" {
			devs = append(devs, strings.Split(a, ",")...)
		} else {
			name = append(name, a)
		}
	}
	entries, err := catalog()
	if err != nil {
		return err
	}
	var e engine.Entry
	switch {
	case len(name) > 0:
		if e, err = resolve(entries, strings.Join(name, " "), chatKinds); err != nil {
			return err
		}
	case s.model != nil:
		e = *s.model
	default:
		return errors.New("usage: /bench [model] [cpu,webgpu,gpu] (no model loaded)")
	}
	if len(devs) == 0 {
		devs = availableDevices()
	}
	reload := s.model
	if reload != nil {
		_ = s.cmdUnload(nil)
	}
	err = benchModel(e, devs, 32, modelFlags{dtype: s.mf.dtype})
	if reload != nil {
		if lerr := s.load(*reload); lerr != nil && err == nil {
			err = lerr
		}
	}
	return err
}
