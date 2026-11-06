// Package coverage_check parses `go test -cover` output and enforces a
// minimum per-package coverage threshold.
package coverage_check

import (
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// coverageLine matches lines like:
//
//	ok  	light/dtype	0.012s	coverage: 95.5% of statements
//	ok  	light/shape	(cached)	coverage: 78.0% of statements
var coverageLine = regexp.MustCompile(`^ok\s+(\S+)\s+.*coverage:\s+([\d.]+)%\s+of\s+statements`)

// noTestLine matches:
//
//	?   	light/nn	[no test files]
var noTestLine = regexp.MustCompile(`^\?\s+(\S+)\s+\[no test files\]`)

// ParseCoverage scans lines from `go test -cover ./...` output and returns a
// map from package path to coverage percent. Packages reported as "no test
// files" are returned with NaN so callers can distinguish "no tests" from
// "zero coverage".
func ParseCoverage(output string) (map[string]float64, error) {
	res := make(map[string]float64)
	for _, line := range splitLines(output) {
		if m := coverageLine.FindStringSubmatch(line); m != nil {
			pct, err := strconv.ParseFloat(m[2], 64)
			if err != nil {
				return nil, fmt.Errorf("parse coverage for %s: %w", m[1], err)
			}
			res[m[1]] = pct
			continue
		}
		if m := noTestLine.FindStringSubmatch(line); m != nil {
			res[m[1]] = nothingToCover
		}
	}
	return res, nil
}

// nothingToCover is a sentinel indicating "no test files". Not NaN so maps
// remain comparable in tests.
const nothingToCover = -1.0

// IsNoTests reports whether a coverage value means "no test files".
func IsNoTests(v float64) bool { return v == nothingToCover }

// EnforceThreshold returns an error listing every package below threshold.
// Packages in skip (exact match) are ignored; packages with no tests are
// ignored as long as their name is in allowEmpty.
func EnforceThreshold(cov map[string]float64, threshold float64, skip, allowEmpty map[string]struct{}) error {
	var failures []string
	pkgs := make([]string, 0, len(cov))
	for k := range cov {
		pkgs = append(pkgs, k)
	}
	sort.Strings(pkgs)
	for _, pkg := range pkgs {
		if _, ok := skip[pkg]; ok {
			continue
		}
		v := cov[pkg]
		if IsNoTests(v) {
			if _, ok := allowEmpty[pkg]; ok {
				continue
			}
			failures = append(failures, fmt.Sprintf("%s: no test files", pkg))
			continue
		}
		if v < threshold {
			failures = append(failures, fmt.Sprintf("%s: %.1f%% < %.1f%%", pkg, v, threshold))
		}
	}
	if len(failures) == 0 {
		return nil
	}
	return fmt.Errorf("coverage below threshold:\n  %s", joinLines(failures))
}

func splitLines(s string) []string {
	var out []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			out = append(out, s[start:i])
			start = i + 1
		}
	}
	if start < len(s) {
		out = append(out, s[start:])
	}
	return out
}

func joinLines(ss []string) string {
	var b strings.Builder
	for i, s := range ss {
		if i > 0 {
			b.WriteString("\n  ")
		}
		b.WriteString(s)
	}
	return b.String()
}

// Ensure errors package is kept for external callers constructing their own.
var _ = errors.New
