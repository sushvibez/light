package coverage_check

import (
	"strings"
	"testing"
)

const sampleOutput = `
ok  	light/dtype	0.012s	coverage: 95.5% of statements
ok  	light/shape	(cached)	coverage: 78.0% of statements
?   	light/nn	[no test files]
ok  	light/storage	0.003s	coverage: 100.0% of statements
?   	light/models	[no test files]
`

func TestParseCoverage(t *testing.T) {
	cov, err := ParseCoverage(sampleOutput)
	if err != nil {
		t.Fatalf("ParseCoverage returned error: %v", err)
	}
	if got, want := cov["light/dtype"], 95.5; got != want {
		t.Errorf("dtype: got %v, want %v", got, want)
	}
	if got, want := cov["light/shape"], 78.0; got != want {
		t.Errorf("shape: got %v, want %v", got, want)
	}
	if got, want := cov["light/storage"], 100.0; got != want {
		t.Errorf("storage: got %v, want %v", got, want)
	}
	if !IsNoTests(cov["light/nn"]) {
		t.Errorf("nn should be no-tests, got %v", cov["light/nn"])
	}
	if !IsNoTests(cov["light/models"]) {
		t.Errorf("models should be no-tests, got %v", cov["light/models"])
	}
	if len(cov) != 5 {
		t.Errorf("expected 5 entries, got %d: %v", len(cov), cov)
	}
}

func TestEnforceThreshold_BelowThreshold(t *testing.T) {
	cov := map[string]float64{
		"light/dtype":   95.5,
		"light/shape":   78.0, // below 80
		"light/storage": 100.0,
	}
	err := EnforceThreshold(cov, 80.0, nil, nil)
	if err == nil {
		t.Fatal("expected error for shape below threshold")
	}
	if !strings.Contains(err.Error(), "light/shape") {
		t.Errorf("error should mention light/shape: %v", err)
	}
	if strings.Contains(err.Error(), "light/dtype") {
		t.Errorf("error must not mention dtype (above threshold): %v", err)
	}
}

func TestEnforceThreshold_AllAbove(t *testing.T) {
	cov := map[string]float64{
		"light/dtype":   95.5,
		"light/storage": 100.0,
	}
	if err := EnforceThreshold(cov, 80.0, nil, nil); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestEnforceThreshold_NoTestsAllowed(t *testing.T) {
	cov := map[string]float64{
		"light/nn":     nothingToCover,
		"light/models": nothingToCover,
	}
	allow := map[string]struct{}{"light/nn": {}, "light/models": {}}
	if err := EnforceThreshold(cov, 80.0, nil, allow); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestEnforceThreshold_NoTestsDisallowed(t *testing.T) {
	cov := map[string]float64{
		"light/nn": nothingToCover,
	}
	err := EnforceThreshold(cov, 80.0, nil, nil)
	if err == nil {
		t.Fatal("expected error when no-tests not allowed")
	}
	if !strings.Contains(err.Error(), "no test files") {
		t.Errorf("error should mention 'no test files': %v", err)
	}
}

func TestEnforceThreshold_Skip(t *testing.T) {
	cov := map[string]float64{
		"light/dtype": 5.0,
	}
	skip := map[string]struct{}{"light/dtype": {}}
	if err := EnforceThreshold(cov, 80.0, skip, nil); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestParseCoverage_CachedFormat(t *testing.T) {
	// `(cached)` token in place of duration should still parse.
	out := "ok  \tlight/dtype\t(cached)\tcoverage: 91.2% of statements\n"
	cov, err := ParseCoverage(out)
	if err != nil {
		t.Fatal(err)
	}
	if cov["light/dtype"] != 91.2 {
		t.Errorf("cached: got %v, want 91.2", cov["light/dtype"])
	}
}
