package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"slices"
	"sort"
	"strings"
	"time"

	"light/device"
	"light/engine"
	"light/internal/cudart"
)

// The performance table: per host (CPU, RAM, OS), per model and device,
// what loading took (time, host RAM, GPU memory) and how fast it ran
// (prefill and decode tokens per second). `light run` and the shell record
// every load and reply; `light bench` measures a standard prompt on each
// device; `light perf` prints it. Stored in ~/.cache/light/perf.json.

type perfEntry struct {
	Model        string    `json:"model"`
	Device       string    `json:"device"` // cpu, webgpu, cuda
	DType        string    `json:"dtype,omitempty"`
	LoadS        float64   `json:"load_s,omitempty"`
	RAMMB        float64   `json:"ram_mb,omitempty"` // host working-set growth across the load
	GPUMB        float64   `json:"gpu_mb,omitempty"` // device memory added by the load
	PromptTokens int       `json:"prompt_tokens,omitempty"`
	PrefillTPS   float64   `json:"prefill_tps,omitempty"`
	DecodeTPS    float64   `json:"decode_tps,omitempty"`
	Source       string    `json:"source,omitempty"` // bench or run
	Updated      time.Time `json:"updated"`
}

type perfHost struct {
	CPU     string                `json:"cpu"`
	RAMGB   float64               `json:"ram_gb"`
	OS      string                `json:"os"`
	GPUs    map[string]string     `json:"gpus,omitempty"` // device kind → adapter / card name
	Entries map[string]*perfEntry `json:"entries"`
}

type perfFile struct {
	Hosts map[string]*perfHost `json:"hosts"`
}

func perfPath() string {
	if p := os.Getenv("LIGHT_PERF_FILE"); p != "" {
		return p
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".cache", "light", "perf.json")
}

// thisHost describes the machine; its key is a hash of CPU, RAM and OS (the
// GPUs are filled in as devices are used, without changing the key).
func thisHost() (string, *perfHost) {
	h := &perfHost{CPU: cpuName(), RAMGB: float64(totalRAM()) / (1 << 30), OS: runtime.GOOS + "/" + runtime.GOARCH}
	sum := sha256.Sum256([]byte(fmt.Sprintf("%s|%.0f|%s", h.CPU, h.RAMGB, h.OS)))
	return hex.EncodeToString(sum[:6]), h
}

func loadPerfFile() *perfFile {
	f := &perfFile{Hosts: map[string]*perfHost{}}
	if b, err := os.ReadFile(perfPath()); err == nil {
		_ = json.Unmarshal(b, f)
		if f.Hosts == nil {
			f.Hosts = map[string]*perfHost{}
		}
	}
	return f
}

// updatePerf applies fn to this host's record and saves the file (write
// to a temporary file, then rename). Failures are reported, not fatal.
func updatePerf(fn func(h *perfHost)) {
	f := loadPerfFile()
	key, fresh := thisHost()
	h := f.Hosts[key]
	if h == nil {
		h = fresh
		f.Hosts[key] = h
	}
	if h.Entries == nil {
		h.Entries = map[string]*perfEntry{}
	}
	if h.GPUs == nil {
		h.GPUs = map[string]string{}
	}
	fn(h)
	b, err := json.MarshalIndent(f, "", "  ")
	if err == nil {
		p := perfPath()
		if err = os.MkdirAll(filepath.Dir(p), 0o755); err == nil {
			tmp := p + ".tmp"
			if err = os.WriteFile(tmp, b, 0o644); err == nil {
				err = os.Rename(tmp, p)
			}
		}
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "light: perf table not saved:", err)
	}
}

func entryFor(h *perfHost, model, dev string) *perfEntry {
	k := model + "|" + dev
	e := h.Entries[k]
	if e == nil {
		e = &perfEntry{Model: model, Device: dev}
		h.Entries[k] = e
	}
	return e
}

// devLabel names a device kind for the table.
func devLabel(d device.Device) string {
	switch d.Kind() {
	case device.KindCUDA:
		return "cuda"
	case device.KindWebGPU:
		return "webgpu"
	}
	return "cpu"
}

// gpuMemMB is the device memory in use, in MB, for the kind the model runs
// on (CUDA: driver used memory; WebGPU: Light's buffers; CPU: 0).
func gpuMemMB(rt *engine.Runtime, kind string) float64 {
	switch kind {
	case "cuda":
		return float64(rt.Status().GPUUsed)
	case "webgpu":
		return webgpuMemMB()
	}
	return 0
}

// gpuName is the adapter / card name of a device kind, if it can be had.
func gpuName(kind string) string {
	switch kind {
	case "cuda":
		if n, err := cudart.DeviceName(0); err == nil {
			return n
		}
	case "webgpu":
		return webgpuName()
	}
	return ""
}

// settleMemory collects garbage until device and host buffers freed through
// runtime cleanups are really gone (a cleanup runs after the GC that found
// its tensor unreachable, so its storage goes at the next one), then
// returns the freed pages to the OS: the working set is then live memory.
func settleMemory() {
	runtime.GC()
	time.Sleep(50 * time.Millisecond) // let queued cleanups run
	debug.FreeOSMemory()
}

// loadMeasured loads spec on rt and records load time, host RAM growth and
// GPU memory growth in the table.
func loadMeasured(rt *engine.Runtime, spec engine.Spec) (*engine.Model, error) {
	kind := spec.Device
	switch kind {
	case "", "gpu":
		kind = "cpu"
		if device.CUDAAvailable() {
			kind = "cuda"
		}
	}
	settleMemory()
	ram0, gpu0 := processRSS(), gpuMemMB(rt, kind)
	m, err := rt.Load(spec)
	if err != nil {
		return nil, err
	}
	settleMemory()
	kind = devLabel(m.Dev)
	ram1, gpu1 := processRSS(), gpuMemMB(rt, kind)
	updatePerf(func(h *perfHost) {
		if n := gpuName(kind); n != "" {
			h.GPUs[kind] = n
		}
		e := entryFor(h, m.Spec.ID, kind)
		e.DType, e.LoadS = m.Spec.DType, m.LoadTime.Seconds()
		e.RAMMB = max(0, float64(int64(ram1)-int64(ram0))/(1<<20))
		e.GPUMB = max(0, gpu1-gpu0)
		e.Updated = time.Now()
		if e.Source == "" {
			e.Source = "run"
		}
	})
	return m, nil
}

// recordReply stores a reply's prefill and decode speeds (replies with too
// few tokens to time are skipped).
func recordReply(m *engine.Model, st engine.Stats, source string) {
	if st.Tokens < 4 || st.PrefillMS <= 0 {
		return
	}
	updatePerf(func(h *perfHost) {
		e := entryFor(h, m.Spec.ID, devLabel(m.Dev))
		if source != "bench" && e.Source == "bench" {
			return // a chat reply's short prompt does not replace the bench numbers
		}
		e.PromptTokens = st.PromptTokens
		e.PrefillTPS = float64(st.PromptTokens) / (st.PrefillMS / 1000)
		e.DecodeTPS = st.TokPerSec
		e.Source, e.Updated = source, time.Now()
	})
}

const benchPrompt = "Write a detailed, multi-paragraph essay about the history of Paris, from its Roman origins as Lutetia through the Middle Ages, the Revolution and Haussmann's renovation, to the city today."

// availableDevices lists the device kinds this build and host can run.
func availableDevices() []string {
	devs := []string{"cpu"}
	if _, err := device.WebGPU(0); err == nil {
		devs = append(devs, "webgpu")
	}
	if device.CUDAAvailable() {
		if _, err := device.CUDA(0); err == nil {
			devs = append(devs, "gpu")
		}
	}
	return devs
}

// benchModel loads e on each device in turn, generates `tokens` greedy
// tokens for the standard prompt, records the numbers and prints them.
func benchModel(e engine.Entry, devices []string, tokens int, mf modelFlags) error {
	for _, dv := range devices {
		mf.device = dv
		spec := mf.spec(e)
		fmt.Fprintf(os.Stderr, "bench %s on %s (%s) …\n", e.Spec.ID, dv, orAuto(spec.DType))
		rt := engine.NewRuntime()
		m, err := loadMeasured(rt, spec)
		if err != nil {
			fmt.Fprintf(os.Stderr, "  %s: %v\n", dv, err)
			continue
		}
		msgs := []engine.Message{{Role: "user", Content: benchPrompt}}
		// Warm up first: the first call pays one-off costs (CUDA JITs the
		// BF16 PTX, WebGPU compiles its pipelines) that would read as a
		// several-times-slower prefill. Nothing carries over between calls.
		_, _, err = rt.Generate(context.Background(), msgs, engine.GenOpts{Temperature: 0, MaxTokens: 2}, nil)
		var st engine.Stats
		if err == nil {
			_, st, err = rt.Generate(context.Background(), msgs, engine.GenOpts{Temperature: 0, MaxTokens: tokens}, nil)
		}
		if err != nil {
			fmt.Fprintf(os.Stderr, "  %s: %v\n", dv, err)
		} else {
			recordReply(m, st, "bench")
			fmt.Fprintf(os.Stderr, "  load %.1f s, prefill %.1f tok/s (%d tokens), decode %.1f tok/s (%d tokens)\n",
				m.LoadTime.Seconds(), float64(st.PromptTokens)/(st.PrefillMS/1000), st.PromptTokens, st.TokPerSec, st.Tokens)
		}
		_ = rt.Unload()
		runtime.GC()
	}
	return printPerf(os.Stdout, e.Spec.ID)
}

func cmdBench(args []string) error {
	fs := flag.NewFlagSet("bench", flag.ContinueOnError)
	devs := fs.String("devices", "", "comma-separated devices to measure: cpu, webgpu, gpu (default: every one this host has)")
	tokens := fs.Int("tokens", 32, "tokens to generate (greedy)")
	dtype := fs.String("dtype", "", "f32 or bf16 for safetensors weights (default: per device)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() != 1 {
		return errors.New("usage: light bench [-devices cpu,webgpu,gpu] [-tokens n] <model>")
	}
	entries, err := catalog()
	if err != nil {
		return err
	}
	e, err := resolve(entries, fs.Arg(0), chatKinds)
	if err != nil {
		return err
	}
	list := availableDevices()
	if *devs != "" {
		list = strings.Split(*devs, ",")
	}
	return benchModel(e, list, *tokens, modelFlags{dtype: *dtype})
}

func cmdPerf(args []string) error {
	fs := flag.NewFlagSet("perf", flag.ContinueOnError)
	asJSON := fs.Bool("json", false, "print every host's table as JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(loadPerfFile())
	}
	filter := ""
	if fs.NArg() > 0 {
		filter = strings.ToLower(fs.Arg(0))
	}
	return printPerf(os.Stdout, filter)
}

// printPerf prints this host's table (rows whose model contains filter).
func printPerf(w *os.File, filter string) error {
	key, fresh := thisHost()
	h := loadPerfFile().Hosts[key]
	fmt.Fprintf(w, "host: %s, %.0f GB RAM, %s\n", fresh.CPU, fresh.RAMGB, fresh.OS)
	if h != nil {
		kinds := make([]string, 0, len(h.GPUs))
		for k := range h.GPUs {
			kinds = append(kinds, k)
		}
		slices.Sort(kinds)
		for _, k := range kinds {
			fmt.Fprintf(w, "%s: %s\n", k, h.GPUs[k])
		}
	}
	var rows []*perfEntry
	if h != nil {
		for _, e := range h.Entries {
			if filter == "" || strings.Contains(strings.ToLower(e.Model), strings.ToLower(filter)) {
				rows = append(rows, e)
			}
		}
	}
	if len(rows) == 0 {
		fmt.Fprintln(w, "no measurements yet: `light bench <model>` measures one on every device")
		return nil
	}
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].Model != rows[j].Model {
			return rows[i].Model < rows[j].Model
		}
		return rows[i].Device < rows[j].Device
	})
	num := func(v float64, f string) string {
		if v <= 0 {
			return "-"
		}
		return fmt.Sprintf(f, v)
	}
	fmt.Fprintf(w, "\n%-44s %-7s %-5s %7s %8s %8s %15s %9s  %s\n", "MODEL", "DEVICE", "DTYPE", "LOAD", "RAM", "GPU MEM", "PREFILL tok/s", "DECODE", "MEASURED")
	for _, e := range rows {
		pre := num(e.PrefillTPS, "%.0f")
		if e.PromptTokens > 0 && pre != "-" {
			pre += fmt.Sprintf(" (%d)", e.PromptTokens)
		}
		fmt.Fprintf(w, "%-44s %-7s %-5s %7s %8s %8s %15s %9s  %s %s\n", trim(e.Model, 44), e.Device, e.DType,
			num(e.LoadS, "%.1f s"), num(e.RAMMB, "%.0f MB"), num(e.GPUMB, "%.0f MB"), pre, num(e.DecodeTPS, "%.1f"),
			e.Updated.Format("2006-01-02"), e.Source)
	}
	fmt.Fprintln(w, "\nRAM: process working-set growth across the load (on an integrated GPU it includes the GPU buffers, which live in system memory).\nPREFILL: prompt tokens per second (prompt length). `light bench <model>` re-measures on every device.")
	return nil
}
