package nn

import (
	"testing"
	"time"

	"light/dtype"
)

// R7: a load in WithDefaultWeightStorage that overlaps a 16-bit scope on
// another goroutine waits for it and sees its own dtypes, never the other
// load's 16-bit storage.
func TestDefaultWeightStorageExcludes16BitScope(t *testing.T) {
	inScope := make(chan struct{})
	release := make(chan struct{})
	scopeDone := make(chan error)
	go func() {
		scopeDone <- WithWeightStorage16(dtype.BF16, func() error {
			if got := WeightStorageDType(dtype.F32, 4); got != dtype.BF16 {
				t.Errorf("inside the 16-bit scope: %v, want BF16", got)
			}
			close(inScope)
			<-release
			return nil
		})
	}()
	<-inScope // the 16-bit scope is now active and held open

	// The leak the lock prevents: an unprotected load sees BF16.
	if got := WeightStorageDType(dtype.F32, 4); got != dtype.BF16 {
		t.Fatalf("precondition: unscoped read during the 16-bit scope gives %v", got)
	}

	plainDone := make(chan dtype.DType)
	go func() {
		var got dtype.DType
		_ = WithDefaultWeightStorage(func() error {
			got = WeightStorageDType(dtype.F32, 4)
			return nil
		})
		plainDone <- got
	}()
	// While the 16-bit scope is held, the plain load must not run.
	select {
	case got := <-plainDone:
		t.Fatalf("plain load ran inside the 16-bit scope (saw %v)", got)
	case <-time.After(50 * time.Millisecond):
	}
	close(release)
	if err := <-scopeDone; err != nil {
		t.Fatal(err)
	}
	if got := <-plainDone; got != dtype.F32 {
		t.Fatalf("plain load saw %v, want F32", got)
	}

	// And the reverse: a 16-bit scope cannot begin inside a plain load.
	inPlain := make(chan struct{})
	releasePlain := make(chan struct{})
	plainErr := make(chan error)
	go func() {
		plainErr <- WithDefaultWeightStorage(func() error {
			close(inPlain)
			<-releasePlain
			if got := WeightStorageDType(dtype.F32, 4); got != dtype.F32 {
				t.Errorf("plain load saw %v after a 16-bit scope was requested", got)
			}
			return nil
		})
	}()
	<-inPlain
	scope16 := make(chan error)
	go func() { scope16 <- WithWeightStorage16(dtype.F16, func() error { return nil }) }()
	select {
	case <-scope16:
		t.Fatal("16-bit scope began inside a plain load")
	case <-time.After(50 * time.Millisecond):
	}
	close(releasePlain)
	if err := <-plainErr; err != nil {
		t.Fatal(err)
	}
	if err := <-scope16; err != nil {
		t.Fatal(err)
	}
}
