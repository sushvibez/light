package nn

import (
	"errors"
	"slices"
	"sync/atomic"
	"testing"
)

// ExpertCache keeps at most Slots experts, drops the least recently used,
// serves repeats from memory, joins a prefetch in flight and counts it,
// and reports a failed load without caching it.
func TestExpertCache(t *testing.T) {
	var reads atomic.Int64
	c, err := NewExpertCache(2, func(e int) (int, error) {
		reads.Add(1)
		if e == 9 {
			return 0, errors.New("bad expert")
		}
		return 10 * e, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	get := func(e int) int {
		v, err := c.Get(e)
		if err != nil {
			t.Fatal(err)
		}
		return v
	}
	if get(1) != 10 || get(2) != 20 || get(1) != 10 {
		t.Fatal("wrong values")
	}
	get(3) // drops 2, the least recently used
	r := c.Resident()
	slices.Sort(r)
	if !slices.Equal(r, []int{1, 3}) {
		t.Fatalf("resident %v, want [1 3]", r)
	}
	if c.Loads != 3 || c.Hits != 1 {
		t.Errorf("%d loads, %d hits; want 3, 1", c.Loads, c.Hits)
	}
	c.Prefetch([]int{4, 5, 6}) // only Slots of them
	if get(4) != 40 || get(5) != 50 {
		t.Fatal("wrong prefetched values")
	}
	if c.Prefetched != 2 || c.PrefetchUsed != 2 {
		t.Errorf("%d prefetched, %d used; want 2, 2", c.Prefetched, c.PrefetchUsed)
	}
	if _, err := c.Get(9); err == nil {
		t.Error("failed load not reported")
	}
	if len(c.Resident()) > 2 {
		t.Errorf("%d resident over 2 slots", len(c.Resident()))
	}
	var seen []int
	if err := c.Each([]int{1, 2, 3, 4}, func(e, v int) error {
		if v != 10*e {
			t.Errorf("expert %d: %d", e, v)
		}
		seen = append(seen, e)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(seen, []int{1, 2, 3, 4}) || c.Prefetched != 2 {
		t.Errorf("Each visited %v (prefetched %d)", seen, c.Prefetched)
	}
}
