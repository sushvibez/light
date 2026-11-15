//go:build !windows && !linux && !darwin && !freebsd && !netbsd && !openbsd && !dragonfly

package hub

import "os"

// No portable file lock here: downloads are still serialized within the
// process by lockDest's mutex.
func lockFile(*os.File) error   { return nil }
func unlockFile(*os.File) error { return nil }
