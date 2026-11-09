package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"light/engine"
	"light/gguf"
	"light/hub"
	"light/models"
)

// chatKinds are the catalog kinds `light run` can chat with.
var chatKinds = map[string]bool{"safetensors": true, "pytorch": true, "gguf": true}

// resolve finds the catalog entry named by q: an exact id, else an exact
// repo or file name (the part after "org/" or "gguf/", ".gguf" optional),
// else the only id containing q, all case-insensitive. kinds limits the
// candidates (nil: any).
func resolve(entries []engine.Entry, q string, kinds map[string]bool) (engine.Entry, error) {
	lq := strings.ToLower(q)
	var cands []engine.Entry
	for _, e := range entries {
		if kinds == nil || kinds[e.Kind] {
			cands = append(cands, e)
		}
	}
	short := func(id string) string {
		id = strings.ToLower(id)
		if i := strings.LastIndex(id, "/"); i >= 0 {
			id = id[i+1:]
		}
		return strings.TrimSuffix(id, ".gguf")
	}
	for _, match := range []func(engine.Entry) bool{
		func(e engine.Entry) bool { return strings.ToLower(e.Spec.ID) == lq },
		func(e engine.Entry) bool { return short(e.Spec.ID) == strings.TrimSuffix(lq, ".gguf") },
		func(e engine.Entry) bool { return strings.Contains(strings.ToLower(e.Spec.ID), lq) },
	} {
		var hits []engine.Entry
		for _, e := range cands {
			if match(e) {
				hits = append(hits, e)
			}
		}
		switch len(hits) {
		case 0:
			continue
		case 1:
			return hits[0], nil
		default:
			ids := make([]string, len(hits))
			for i, h := range hits {
				ids[i] = "  " + h.Spec.ID
			}
			return engine.Entry{}, fmt.Errorf("%q matches %d models; be more specific:\n%s", q, len(hits), strings.Join(ids, "\n"))
		}
	}
	return engine.Entry{}, fmt.Errorf("no model matches %q (see `light list`; `light pull <org/repo>` downloads one)", q)
}

func catalog() ([]engine.Entry, error) {
	entries, err := engine.Catalog()
	if err != nil {
		return nil, fmt.Errorf("scanning models: %w", err)
	}
	return entries, nil
}

// gb formats a byte count for humans (GB, MB or KB).
func gb(n int64) string {
	switch {
	case n >= 1<<30:
		return fmt.Sprintf("%.1f GB", float64(n)/(1<<30))
	case n >= 1<<20:
		return fmt.Sprintf("%.0f MB", float64(n)/(1<<20))
	default:
		return fmt.Sprintf("%.0f KB", float64(n)/(1<<10))
	}
}

func cmdList(args []string) error {
	fs := flag.NewFlagSet("list", flag.ContinueOnError)
	asJSON := fs.Bool("json", false, "print the catalog as JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}
	entries, err := catalog()
	if err != nil {
		return err
	}
	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(entries)
	}
	if len(entries) == 0 {
		fmt.Println("no models yet; `light pull <org/repo>` downloads one")
		return nil
	}
	fmt.Printf("%-58s %-11s %-14s %6s %9s\n", "MODEL", "KIND", "TYPE", "PARAMS", "SIZE")
	for _, e := range entries {
		fmt.Printf("%-58s %-11s %-14s %6s %9s\n", e.Spec.ID, e.Kind, e.ModelType, e.Params, gb(e.SizeBytes))
	}
	return nil
}

// repoDir is the HF cache directory of repo.
func repoDir(repo string) (string, error) {
	root, err := hub.CacheDir()
	if err != nil {
		return "", err
	}
	return filepath.Join(root, "models--"+strings.ReplaceAll(repo, "/", "--")), nil
}

func cmdInfo(args []string) error {
	fs := flag.NewFlagSet("info", flag.ContinueOnError)
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 1 {
		return errors.New("usage: light info <model>")
	}
	entries, err := catalog()
	if err != nil {
		return err
	}
	e, err := resolve(entries, fs.Arg(0), nil)
	if err != nil {
		return err
	}
	fmt.Printf("model    %s\nkind     %s\ntype     %s\nparams   %s\nsize     %s\nvocab    %d\ndtype    %s\n",
		e.Spec.ID, e.Kind, e.ModelType, e.Params, gb(e.SizeBytes), e.VocabSize, e.Spec.DType)
	if e.Spec.GGUF != "" {
		fmt.Printf("file     %s\n", e.Spec.GGUF)
		return ggufInfo(e.Spec.GGUF)
	}
	if e.Spec.Repo != "" {
		dir, err := repoDir(e.Spec.Repo)
		if err != nil {
			return err
		}
		snaps, _ := filepath.Glob(filepath.Join(dir, "snapshots", "*", "config.json"))
		if len(snaps) > 0 {
			fmt.Printf("files    %s\n", filepath.Dir(snaps[len(snaps)-1]))
			return configInfo(snaps[len(snaps)-1])
		}
	}
	return nil
}

// configInfo prints the architecture fields of a config.json (top level,
// or under text_config for multimodal configs).
func configInfo(p string) error {
	raw, err := os.ReadFile(p)
	if err != nil {
		return err
	}
	var cfg map[string]any
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return err
	}
	if tc, ok := cfg["text_config"].(map[string]any); ok {
		for k, v := range tc {
			if _, dup := cfg[k]; !dup {
				cfg[k] = v
			}
		}
	}
	for _, k := range []struct{ key, label string }{
		{"architectures", "arch"}, {"hidden_size", "hidden"}, {"num_hidden_layers", "layers"},
		{"num_attention_heads", "heads"}, {"num_key_value_heads", "kv heads"}, {"num_experts", "experts"},
		{"num_local_experts", "experts"}, {"n_routed_experts", "experts"}, {"max_position_embeddings", "context"},
		{"torch_dtype", "weights"}, {"dtype", "weights"},
	} {
		if v, ok := cfg[k.key]; ok && v != nil {
			fmt.Printf("%-8s %v\n", k.label, v)
		}
	}
	if q, ok := cfg["quantization_config"].(map[string]any); ok {
		fmt.Printf("quant    %v %v-bit\n", q["quant_method"], q["bits"])
	}
	return nil
}

func ggufInfo(p string) error {
	f, err := os.Open(p)
	if err != nil {
		return err
	}
	defer f.Close()
	g, err := gguf.Open(f)
	if err != nil {
		return err
	}
	arch, _ := g.String("general.architecture")
	fmt.Printf("arch     %s\n", arch)
	for _, k := range []struct{ key, label string }{
		{"embedding_length", "hidden"}, {"block_count", "layers"}, {"attention.head_count", "heads"},
		{"attention.head_count_kv", "kv heads"}, {"expert_count", "experts"}, {"context_length", "context"},
	} {
		if v, ok := g.Int64(arch + "." + k.key); ok {
			fmt.Printf("%-8s %d\n", k.label, v)
		}
	}
	if ft, ok := g.Int64("general.file_type"); ok {
		fmt.Printf("ftype    %d\n", ft)
	}
	fmt.Printf("tensors  %d\n", len(g.Tensors))
	return nil
}

func cmdRm(args []string) error {
	fs := flag.NewFlagSet("rm", flag.ContinueOnError)
	yes := fs.Bool("y", false, "do not ask for confirmation")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 1 {
		return errors.New("usage: light rm [-y] <model>")
	}
	entries, err := catalog()
	if err != nil {
		return err
	}
	e, err := resolve(entries, fs.Arg(0), nil)
	if err != nil {
		return err
	}
	var targets []string
	if e.Kind == "gguf" {
		targets = append(targets, e.Spec.GGUF)
		if e.Spec.TokenizerPath != "" {
			targets = append(targets, e.Spec.TokenizerPath)
		}
	} else {
		dir, err := repoDir(e.Spec.Repo)
		if err != nil {
			return err
		}
		targets = append(targets, dir)
	}
	if !*yes {
		fmt.Printf("delete %s (%s)?\n  %s\n[y/N] ", e.Spec.ID, gb(e.SizeBytes), strings.Join(targets, "\n  "))
		line, _ := bufio.NewReader(os.Stdin).ReadString('\n')
		if a := strings.ToLower(strings.TrimSpace(line)); a != "y" && a != "yes" {
			fmt.Println("kept")
			return nil
		}
	}
	for _, t := range targets {
		if err := os.RemoveAll(t); err != nil {
			return err
		}
	}
	fmt.Println("deleted", e.Spec.ID)
	return nil
}

// pullPlan is what `light pull` fetches from a repo listing: files for the
// HF cache, files for the GGUF directory, or (choices) several GGUF
// quantizations to pick from.
type pullPlan struct {
	cache, ggufDir, choices []hub.RepoFile
}

// planPull chooses the files of a whole-repo pull. Safetensors checkpoints
// get their top-level weights, configs and tokenizer files (not onnx /
// flax / tf exports or subfolders); a repo with no safetensors but one
// GGUF gets that file in the GGUF directory (several: the caller asks
// which); PyTorch .bin weights are the last resort. Moshi's repo keeps its
// GGUF in the HF cache, where the speech catalog looks.
func planPull(repo string, files []hub.RepoFile) pullPlan {
	var p pullPlan
	topLevel := func(f hub.RepoFile) bool { return !strings.Contains(f.Name, "/") }
	has := func(pat string) bool {
		for _, f := range files {
			if ok, _ := path.Match(pat, f.Name); ok && topLevel(f) {
				return true
			}
		}
		return false
	}
	aux := func(name string) bool {
		base := strings.ToLower(name)
		return strings.HasSuffix(base, ".json") || strings.HasSuffix(base, ".jinja") || strings.HasSuffix(base, ".tiktoken") ||
			base == "tokenizer.model" || base == "merges.txt" || strings.HasPrefix(base, "vocab.")
	}
	if repo == models.MoshikoRepo {
		for _, f := range files {
			if f.Name == models.MoshikoGGUF || f.Name == models.MoshikoTokenizer {
				p.cache = append(p.cache, f)
			}
		}
		return p
	}
	var ggufs []hub.RepoFile
	for _, f := range files {
		if strings.HasSuffix(strings.ToLower(f.Name), ".gguf") {
			ggufs = append(ggufs, f)
		}
	}
	weights := ""
	switch {
	case has("*.safetensors"):
		weights = "*.safetensors"
	case len(ggufs) == 1:
		p.ggufDir = ggufs
		return p
	case len(ggufs) > 1:
		p.choices = ggufs
		return p
	case has("pytorch_model*.bin"):
		weights = "pytorch_model*.bin"
	}
	// Mistral's repos also ship consolidated.safetensors (its own format, the
	// same weights again): skip it when HF-layout shards are present.
	hfShards := has("model*.safetensors")
	for _, f := range files {
		if !topLevel(f) || (hfShards && strings.HasPrefix(f.Name, "consolidated")) {
			continue
		}
		if ok, _ := path.Match(weights, f.Name); (ok && weights != "") || aux(f.Name) {
			p.cache = append(p.cache, f)
		}
	}
	return p
}

func cmdPull(args []string) error {
	fs := flag.NewFlagSet("pull", flag.ContinueOnError)
	rev := fs.String("revision", "main", "branch, tag or commit")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 || strings.Count(fs.Arg(0), "/") != 1 {
		return errors.New("usage: light pull <org/repo> [file ...]")
	}
	repo := fs.Arg(0)
	files, err := hub.ListFiles(repo, *rev)
	if err != nil {
		return err
	}
	var plan pullPlan
	if fs.NArg() > 1 {
		byName := map[string]hub.RepoFile{}
		for _, f := range files {
			byName[f.Name] = f
		}
		for _, name := range fs.Args()[1:] {
			f, ok := byName[name]
			if !ok {
				return fmt.Errorf("%s has no file %q", repo, name)
			}
			if strings.HasSuffix(strings.ToLower(name), ".gguf") && repo != models.MoshikoRepo {
				plan.ggufDir = append(plan.ggufDir, f)
			} else {
				plan.cache = append(plan.cache, f)
			}
		}
	} else {
		plan = planPull(repo, files)
	}
	if len(plan.choices) > 0 {
		sort.Slice(plan.choices, func(i, j int) bool { return plan.choices[i].Name < plan.choices[j].Name })
		fmt.Printf("%s has %d GGUF files; pick one:\n", repo, len(plan.choices))
		for _, f := range plan.choices {
			fmt.Printf("  light pull %s %-50s %9s\n", repo, f.Name, gb(f.Size))
		}
		return nil
	}
	if len(plan.cache)+len(plan.ggufDir) == 0 {
		return fmt.Errorf("%s has no safetensors, GGUF or PyTorch weights", repo)
	}
	var total int64
	for _, f := range append(append([]hub.RepoFile{}, plan.cache...), plan.ggufDir...) {
		total += f.Size
	}
	fmt.Printf("pulling %s: %d files, %s\n", repo, len(plan.cache)+len(plan.ggufDir), gb(total))
	// Small files (configs, tokenizer) first, so an interrupted pull still
	// leaves a usable model directory to resume; each file is retried,
	// resuming its .part, before the pull gives up on it.
	bySize := func(fs []hub.RepoFile) {
		sort.SliceStable(fs, func(i, j int) bool { return fs[i].Size < fs[j].Size })
	}
	bySize(plan.cache)
	bySize(plan.ggufDir)
	var failed []string
	fetch := func(f hub.RepoFile, get func() error) {
		var err error
		for attempt := 1; attempt <= pullAttempts; attempt++ {
			if err = get(); err == nil {
				doneBar(f)
				return
			}
			fmt.Fprintf(os.Stderr, "\n  %s: %v (attempt %d of %d)\n", f.Name, err, attempt, pullAttempts)
			time.Sleep(time.Duration(attempt) * 2 * time.Second)
		}
		failed = append(failed, f.Name)
	}
	for _, f := range plan.cache {
		fetch(f, func() error { _, err := hub.DownloadWithProgress(repo, *rev, f.Name, newBar(f)); return err })
	}
	for _, f := range plan.ggufDir {
		dst := filepath.Join(engine.GGUFDir(), path.Base(f.Name))
		fetch(f, func() error { return hub.DownloadTo(repo, *rev, f.Name, dst, newBar(f)) })
	}
	if len(failed) > 0 {
		return fmt.Errorf("%d file(s) failed after %d attempts each: %s (run the same pull again to resume)", len(failed), pullAttempts, strings.Join(failed, ", "))
	}
	fmt.Println("done; `light list` shows it")
	return nil
}

// pullAttempts is how often a file download is tried (each retry resumes).
const pullAttempts = 4

// newBar returns a progress callback drawing one line for f on stderr.
func newBar(f hub.RepoFile) hub.Progress {
	start, last := time.Now(), time.Time{}
	return func(done, total int64) {
		if total == 0 {
			total = f.Size
		}
		if now := time.Now(); now.Sub(last) < 200*time.Millisecond && done != total {
			return
		} else {
			last = now
		}
		pct := 0.0
		if total > 0 {
			pct = 100 * float64(done) / float64(total)
		}
		rate := float64(done) / max(time.Since(start).Seconds(), 1e-3) / (1 << 20)
		fmt.Fprintf(os.Stderr, "\r  %-48s %8s / %-8s %5.1f%% %7.1f MB/s", trim(f.Name, 48), gb(done), gb(total), pct, rate)
	}
}

func doneBar(f hub.RepoFile) {
	fmt.Fprintf(os.Stderr, "\r  %-48s %8s %-30s\n", trim(f.Name, 48), gb(f.Size), "ok")
}

func trim(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return "…" + s[len(s)-n+1:]
}
