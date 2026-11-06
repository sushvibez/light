package registry_test

import (
	"errors"
	"testing"

	"light/device"
	"light/dtype"
	"light/internal/registry"
)

func TestRegisterAndLookup(t *testing.T) {
	key := registry.Key{Op: "_test_add", Dev: device.KindCPU, DType: dtype.F32}
	called := false
	registry.Register(key, func(ctx *registry.Context) error {
		called = true
		return nil
	})
	fn, err := registry.Lookup(key)
	if err != nil {
		t.Fatal(err)
	}
	if err := fn(nil); err != nil {
		t.Errorf("kernel returned error: %v", err)
	}
	if !called {
		t.Error("kernel not invoked")
	}
}

func TestLookupNotFound(t *testing.T) {
	_, err := registry.Lookup(registry.Key{Op: "_nonexistent", Dev: device.KindCPU, DType: dtype.F32})
	if !errors.Is(err, registry.ErrKernelNotFound) {
		t.Errorf("expected ErrKernelNotFound, got %v", err)
	}
}

func TestRegisterDuplicate(t *testing.T) {
	key := registry.Key{Op: "_test_dup", Dev: device.KindCPU, DType: dtype.F32}
	registry.Register(key, func(ctx *registry.Context) error { return nil })
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on duplicate register")
		}
	}()
	registry.Register(key, func(ctx *registry.Context) error { return nil })
}

func TestKeyString(t *testing.T) {
	k := registry.Key{Op: "add", Dev: device.KindCPU, DType: dtype.F32}
	s := k.String()
	if s == "" {
		t.Error("empty string")
	}
}

func TestRegistered(t *testing.T) {
	// At least our own test registrations should be present.
	keys := registry.Registered()
	found := false
	for _, k := range keys {
		if k.Op == "_test_add" {
			found = true
			break
		}
	}
	if !found {
		t.Error("Registered() should include _test_add")
	}
}
