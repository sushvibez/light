package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"time"

	"light/engine"
	"light/models"
	"light/tokenizer"
)

// cmdEnergraft continues a prompt with an Energraft-EBT release: each next
// token is a Gaussian candidate refined by the energy head's gradient.
func cmdEnergraft(args []string) error {
	fs := flag.NewFlagSet("energraft", flag.ContinueOnError)
	dev := fs.String("device", "", "cpu, webgpu or gpu (default: CUDA when present, else cpu)")
	repo := fs.String("repo", "dimweak/Energraft-EBT-Qwen3-1.7B-Base", "the release: a Hub repo or a folder")
	n := fs.Int("max-tokens", 64, "tokens to generate")
	steps := fs.Int("steps", 0, "energy updates per token (0: the release's, 2)")
	seed := fs.Uint64("seed", 1234, "candidate noise seed")
	fs.Usage = func() {
		fmt.Fprintln(os.Stderr, "usage: light energraft [flags] <prompt>\n\nA base (not chat) energy-based LM: it continues the prompt text.\n\nflags:")
		fs.PrintDefaults()
	}
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 {
		fs.Usage()
		return fmt.Errorf("want a prompt")
	}
	prompt := strings.Join(fs.Args(), " ")
	d, err := engine.PickDevice(*dev)
	if err != nil {
		return err
	}
	t0 := time.Now()
	m, err := models.LoadEnergraft(*repo, d)
	if err != nil {
		return err
	}
	tok, err := tokenizer.Load(filepath.Join(m.Dir, "tokenizer.json"))
	if err != nil {
		return err
	}
	eos, err := energraftEOS(filepath.Join(m.Dir, "config.json"))
	if err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "loaded in %.1fs on %v\n", time.Since(t0).Seconds(), d)
	var ids []int64
	for _, id := range tok.Encode(prompt) {
		ids = append(ids, int64(id))
	}
	fmt.Print(prompt)
	var gen []int32
	printed := ""
	t1 := time.Now()
	out, err := m.Generate(ids, models.EnergraftOpts{MaxTokens: *n, Steps: *steps, EOS: eos,
		Rng: rand.New(rand.NewPCG(*seed, *seed^0x9E3779B97F4A7C15)),
		OnToken: func(id int64) bool {
			if eos[id] {
				return true
			}
			gen = append(gen, int32(id))
			text := tok.Decode(gen)
			if strings.HasPrefix(text, printed) && !strings.HasSuffix(text, "�") {
				fmt.Print(text[len(printed):])
				printed = text
			}
			return true
		}})
	fmt.Println()
	if len(out) > 0 {
		fmt.Fprintf(os.Stderr, "%d tokens, %.2f s/token\n", len(out), time.Since(t1).Seconds()/float64(len(out)))
	}
	return err
}

// energraftEOS reads config.json's eos_token_id (an id or a list).
func energraftEOS(path string) (map[int64]bool, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var c struct {
		EOS json.RawMessage `json:"eos_token_id"`
	}
	if err := json.Unmarshal(raw, &c); err != nil {
		return nil, err
	}
	out := map[int64]bool{}
	var one int64
	var many []int64
	if json.Unmarshal(c.EOS, &one) == nil {
		out[one] = true
	} else if json.Unmarshal(c.EOS, &many) == nil {
		for _, id := range many {
			out[id] = true
		}
	}
	return out, nil
}
