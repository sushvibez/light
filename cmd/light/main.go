// light is the command-line front end to the Light engine: chat with a
// model in the terminal, manage the models on disk, and serve them through
// an OpenAI-compatible API.
//
//	light                              interactive shell: chat, /model, /pull, /help …
//	light list                         models on disk (HF cache + GGUF dir)
//	light pull Qwen/Qwen2.5-0.5B-Instruct
//	light pull unsloth/Qwen3-4B-GGUF Qwen3-4B-Q4_K_M.gguf
//	light run qwen2.5-0.5b             interactive chat
//	light run qwen2.5-0.5b "Why is the sky blue?"
//	echo "Summarise: ..." | light run qwen2.5-0.5b
//	light info qwen2.5-0.5b
//	light rm qwen2.5-0.5b
//	light serve -model qwen2.5-0.5b    OpenAI-compatible API on :8129
//
// Models are named by catalog id (`light list`) or any unique part of one.
package main

import (
	"fmt"
	"os"
	"runtime/debug"
)

type command struct {
	name, args, help string
	run              func(args []string) error
}

var commands []command

func init() {
	commands = []command{
		{"list", "[-json]", "list the models on disk", cmdList},
		{"pull", "<org/repo> [file.gguf ...]", "download a model from the Hugging Face Hub", cmdPull},
		{"run", "<model> [prompt]", "chat in the terminal, or answer one prompt (argument or stdin)", cmdRun},
		{"info", "<model>", "show a model's architecture and files", cmdInfo},
		{"rm", "<model>", "delete a model from disk", cmdRm},
		{"serve", "[-model <model>]", "serve models through an OpenAI-compatible HTTP API", cmdServe},
		{"bench", "<model>", "measure load / RAM / GPU memory / prefill / decode on each device", cmdBench},
		{"perf", "[model]", "this host's measured performance table", cmdPerf},
		{"image", "<org/repo> <prompt>", "generate an image (FLUX.2 klein, FLUX.1) into a PNG", cmdImage},
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: light                      interactive shell (/help inside)\n       light <command> [arguments]\n\ncommands:")
	for _, c := range commands {
		fmt.Fprintf(os.Stderr, "  %-6s %-30s %s\n", c.name, c.args, c.help)
	}
	fmt.Fprintln(os.Stderr, "\nRun `light <command> -h` for a command's flags.")
}

func main() {
	// Model weights are long-lived, pointer-free buffers on the Go heap; the
	// default GOGC=100 lets the heap grow to twice them before collecting
	// (a 2 GB model ran at ~4.5 GB). Marking them costs almost nothing, so
	// collect at 25 % growth instead ($GOGC still overrides).
	if os.Getenv("GOGC") == "" {
		debug.SetGCPercent(25)
	}
	if len(os.Args) == 1 {
		if st, err := os.Stdin.Stat(); err == nil && st.Mode()&os.ModeCharDevice != 0 {
			if err := cmdShell(); err != nil {
				fmt.Fprintln(os.Stderr, "light:", err)
				os.Exit(1)
			}
			return
		}
	}
	if len(os.Args) < 2 || os.Args[1] == "-h" || os.Args[1] == "--help" || os.Args[1] == "help" {
		usage()
		if len(os.Args) < 2 {
			os.Exit(2)
		}
		return
	}
	for _, c := range commands {
		if c.name == os.Args[1] {
			if err := c.run(os.Args[2:]); err != nil {
				fmt.Fprintln(os.Stderr, "light "+c.name+":", err)
				os.Exit(1)
			}
			return
		}
	}
	fmt.Fprintf(os.Stderr, "light: unknown command %q\n\n", os.Args[1])
	usage()
	os.Exit(2)
}
