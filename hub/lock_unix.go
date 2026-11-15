//go:build linux || darwin || freebsd || netbsd || openbsd || dragonfly

package hub

import (
	"os"
	"syscall"
)

// lockFile blocks until it holds an exclusive flock on f.
func lockFile(f *os.File) error {
	for {
		err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX)
		if err != syscall.EINTR {
			return err
		}
	}
}

func unlockFile(f *os.File) error { return syscall.Flock(int(f.Fd()), syscall.LOCK_UN) }
