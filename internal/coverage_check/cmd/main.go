// Command coverage_check runs `go test -cover ./...` and fails if any
// package is below the threshold, subject to skip / allow-empty lists.
//
// Usage:
//
//	go run ./internal/coverage_check/cmd -threshold 80 -tags "" -skip "pkg1,pkg2" -allow-empty "pkg3"
package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"strings"

	cc "light/internal/coverage_check"
)

func main() {
	threshold := flag.Float64("threshold", 80.0, "minimum coverage percent per package")
	tags := flag.String("tags", "", "build tags for go test (e.g. 'cuda')")
	pkgs := flag.String("pkgs", "./...", "packages to test")
	skipCSV := flag.String("skip", "", "comma-separated packages to skip")
	allowCSV := flag.String("allow-empty", "", "comma-separated packages allowed to have no tests")
	flag.Parse()

	args := []string{"test", "-cover", "-count=1"}
	if *tags != "" {
		args = append(args, "-tags", *tags)
	}
	args = append(args, strings.Split(*pkgs, " ")...)

	cmd := exec.Command("go", args...)
	cmd.Stderr = os.Stderr
	out, err := cmd.Output()
	if err != nil {
		fmt.Fprintln(os.Stderr, "go test failed:", err)
		os.Exit(2)
	}

	cov, err := cc.ParseCoverage(string(out))
	if err != nil {
		fmt.Fprintln(os.Stderr, "parse coverage:", err)
		os.Exit(2)
	}

	skip := toSet(*skipCSV)
	// Always skip the cmd package itself — it's a thin CLI wrapper.
	skip["light/internal/coverage_check/cmd"] = struct{}{}
	allowEmpty := toSet(*allowCSV)
	// Packages consisting of a single doc.go (or similar) legitimately have
	// no tests; we'll pre-populate allowEmpty for common cases. Callers can
	// override.
	if len(allowEmpty) == 0 {
		allowEmpty = defaultAllowEmpty()
	}

	if err := cc.EnforceThreshold(cov, *threshold, skip, allowEmpty); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("coverage OK")
}

func toSet(csv string) map[string]struct{} {
	m := make(map[string]struct{})
	if csv == "" {
		return m
	}
	for k := range strings.SplitSeq(csv, ",") {
		k = strings.TrimSpace(k)
		if k != "" {
			m[k] = struct{}{}
		}
	}
	return m
}

// defaultAllowEmpty lists packages that are scaffolding-only at early phases
// and do not yet require tests. This list should shrink over time.
func defaultAllowEmpty() map[string]struct{} {
	return map[string]struct{}{
		"light/autograd":          {},
		"light/device":            {},
		"light/dtype":             {},
		"light/gguf":              {},
		"light/hub":               {},
		"light/internal/cudart":   {},
		"light/internal/registry": {},
		"light/kernels/cpu":       {},
		"light/kernels/cuda":      {},
		"light/models":            {},
		"light/nn":                {},
		"light/ops":               {},
		"light/quant":             {},
		"light/safetensors":       {},
		"light/shape":             {},
		"light/storage":           {},
		"light/tensor":            {},
		"light/tokenizer":         {},
		"light/training":          {},
		"light/transformers":      {},
	}
}
