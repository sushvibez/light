package hub

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

// destLocks serializes downloads of one destination within the process;
// lockFile (flock / LockFileEx) extends that across processes sharing a
// cache. OS locks die with their process, so a crash never leaves a stale
// lock behind. Each in-process lock is a one-slot channel so a wait can be
// abandoned when the caller's context ends.
var (
	destLocksMu sync.Mutex
	destLocks   = map[string]chan struct{}{}
)

// lockPollMax bounds the back-off between attempts on a cross-process lock
// another process holds.
const lockPollMax = 500 * time.Millisecond

// lockDest takes exclusive ownership of dst (and its dst+".part") for the
// whole check/resume/download/publish sequence. The returned func releases
// it. Waiting (on another goroutine or another process) ends with ctx's
// error when ctx is done first.
func lockDest(ctx context.Context, dst string) (func(), error) {
	key := dst
	if abs, err := filepath.Abs(dst); err == nil {
		key = abs
	}
	if runtime.GOOS == "windows" || runtime.GOOS == "darwin" {
		key = strings.ToLower(key) // case-insensitive filesystems
	}
	destLocksMu.Lock()
	sem := destLocks[key]
	if sem == nil {
		sem = make(chan struct{}, 1)
		destLocks[key] = sem
	}
	destLocksMu.Unlock()
	select {
	case sem <- struct{}{}:
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	release := func() { <-sem }

	f, err := openLockFile(key, dst)
	if err != nil {
		release()
		return nil, err
	}
	wait := 10 * time.Millisecond
	for {
		ok, err := tryLockFile(f)
		if err != nil {
			_ = f.Close()
			release()
			return nil, err
		}
		if ok {
			break
		}
		t := time.NewTimer(wait)
		select {
		case <-t.C:
		case <-ctx.Done():
			t.Stop()
			_ = f.Close()
			release()
			return nil, ctx.Err()
		}
		wait = min(2*wait, lockPollMax)
	}
	return func() {
		_ = unlockFile(f)
		_ = f.Close()
		release()
	}, nil
}

// openLockFile opens the cross-process lock for dst. Lock files live in
// <cache>/.locks (as huggingface_hub keeps its own), named by a hash of the
// destination, so snapshot directories stay clean; without a usable cache
// dir they sit next to the destination. They are never deleted: removing a
// lock file another process has open would let a third process lock a
// fresh inode concurrently.
func openLockFile(key, dst string) (*os.File, error) {
	sum := sha256.Sum256([]byte(key))
	name := "light-" + hex.EncodeToString(sum[:12]) + ".lock"
	if cache, err := CacheDir(); err == nil {
		dir := filepath.Join(cache, ".locks")
		if err := os.MkdirAll(dir, 0o755); err == nil {
			if f, err := os.OpenFile(filepath.Join(dir, name), os.O_RDWR|os.O_CREATE, 0o644); err == nil {
				return f, nil
			}
		}
	}
	return os.OpenFile(dst+".lock", os.O_RDWR|os.O_CREATE, 0o644)
}
