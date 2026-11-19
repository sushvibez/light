//go:build linux || darwin || freebsd || netbsd || openbsd || dragonfly

package hub

import (
	"os"
	"syscall"
)

// tryLockFile takes an exclusive flock on f without blocking; ok is false
// when another process holds it.
func tryLockFile(f *os.File) (bool, error) {
	for {
		err := syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
		switch err {
		case nil:
			return true, nil
		case syscall.EWOULDBLOCK:
			return false, nil
		case syscall.EINTR:
			continue
		}
		return false, err
	}
}

func unlockFile(f *os.File) error { return syscall.Flock(int(f.Fd()), syscall.LOCK_UN) }
