//go:build !nocuda && cgo

package ops_test

import (
	"os"
	"sort"
	"testing"
	"time"

	"light/device"
	"light/dtype"
	"light/internal/cudart"
	"light/ops"
	"light/storage"
	"light/tensor"
)

// TestAllocJitterCUDA (LIGHT_OPBENCH=1) times one elementwise op call by
// call and reports the spread next to the allocator's slow paths, to see
// what a pool miss costs.
func TestAllocJitterCUDA(t *testing.T) {
	if os.Getenv("LIGHT_OPBENCH") == "" {
		t.Skip("set LIGHT_OPBENCH=1")
	}
	gpu, err := device.CUDA(0)
	if err != nil {
		t.Skip(err)
	}
	x, _ := tensor.Ones(dtype.F32, []int{1, 4352, 3072}, gpu)
	run := func(name string, n int) {
		before := storage.CUDAPoolStats()
		var ds []time.Duration
		for range n {
			_ = cudart.DeviceSynchronize()
			t0 := time.Now()
			if _, err := ops.Add(x, x); err != nil {
				t.Fatal(err)
			}
			_ = cudart.DeviceSynchronize()
			ds = append(ds, time.Since(t0))
		}
		after := storage.CUDAPoolStats()
		sorted := append([]time.Duration(nil), ds...)
		sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
		t.Logf("%s: %d calls: min %v, median %v, p90 %v, max %v; first 12: %v", name, n, sorted[0], sorted[n/2], sorted[n*9/10], sorted[n-1], ds[:12])
		t.Logf("%s: mallocs %d (%v), best fits %d, forced GCs %d (%v), evictions %d (%v)", name,
			after.Mallocs-before.Mallocs, after.MallocTime-before.MallocTime, after.BestFits-before.BestFits,
			after.ForcedGCs-before.ForcedGCs, after.GCTime-before.GCTime, after.Evictions-before.Evictions, after.EvictTime-before.EvictTime)
	}
	run("cold", 60)
	run("warm", 60)
}
