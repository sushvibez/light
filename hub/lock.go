package hub

import (
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
)

// destLocks serializes downloads of one destination within the process;
// lockFile (flock / LockFileEx) extends that across processes sharing a
// cache. OS locks die with their process, so a crash never leaves a stale
// lock behind.
var (
	destLocksMu sync.Mutex
	destLocks   = map[string]*sync.Mutex{}
)

// lockDest takes exclusive ownership of dst (and its dst+".part") for the
// whole check/resume/download/publish sequence. The returned func releases
// it.
func lockDest(dst string) (func(), error) {
	key := dst
	if abs, err := filepath.Abs(dst); err == nil {
		key = abs
	}
	if runtime.GOOS == "windows" || runtime.GOOS == "darwin" {
		key = strings.ToLower(key) // case-insensitive filesystems
	}
	destLocksMu.Lock()
	mu := destLocks[key]
	if mu == nil {
		mu = new(sync.Mutex)
		destLocks[key] = mu
	}
	destLocksMu.Unlock()
	mu.Lock()

	f, err := openLockFile(key, dst)
	if err != nil {
		mu.Unlock()
		return nil, err
	}
	if err := lockFile(f); err != nil {
		_ = f.Close()
		mu.Unlock()
		return nil, err
	}
	return func() {
		_ = unlockFile(f)
		_ = f.Close()
		mu.Unlock()
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
