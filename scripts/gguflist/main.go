// gguflist prints every metadata key and every tensor (dtype, dims) of a
// GGUF file, reading only the header, so it also works on a partially
// downloaded file. Tensor names are collapsed by replacing numeric path
// components with N when -collapse is given.
//
//	go run ./scripts/gguflist [-collapse] model.gguf
package main

import (
	"flag"
	"fmt"
	"os"
	"regexp"
	"sort"

	"light/gguf"
)

func main() {
	collapse := flag.Bool("collapse", false, "collapse numeric path components to N and print each pattern once")
	flag.Parse()
	if flag.NArg() != 1 {
		fmt.Fprintln(os.Stderr, "usage: gguflist [-collapse] model.gguf")
		os.Exit(2)
	}
	f, err := os.Open(flag.Arg(0))
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer f.Close()
	g, err := gguf.Open(f)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	keys := make([]string, 0, len(g.Meta))
	for k := range g.Meta {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		v := g.Meta[k]
		if s, ok := v.(string); ok && len(s) > 80 {
			v = s[:80] + "…"
		}
		if arr, ok := v.([]any); ok && len(arr) > 8 {
			v = fmt.Sprintf("[%d values] %v…", len(arr), arr[:8])
		}
		fmt.Printf("KV %s = %v\n", k, v)
	}
	names := make([]string, 0, len(g.Tensors))
	for n := range g.Tensors {
		names = append(names, n)
	}
	sort.Strings(names)
	num := regexp.MustCompile(`\.\d+\.`)
	seen := map[string]bool{}
	for _, n := range names {
		t := g.Tensors[n]
		key := n
		if *collapse {
			key = num.ReplaceAllString(n, ".N.")
			if seen[key] {
				continue
			}
			seen[key] = true
		}
		fmt.Printf("%-8v %-24v %s\n", t.DType, t.Dims, n)
	}
	fmt.Printf("%d tensors\n", len(g.Tensors))
}
