//go:build !windows && !linux && !darwin && !freebsd && !netbsd && !openbsd && !dragonfly

package hub

import "os"

// No portable file lock here: downloads are still serialized within the
// process by lockDest's in-process lock.
func tryLockFile(*os.File) (bool, error) { return true, nil }
func unlockFile(*os.File) error          { return nil }
