// Package hostmem reports the host's available physical memory, for tests
// that must not start a load the machine cannot hold.
package hostmem

// AvailableGB is the physical memory available to new allocations in GiB
// (Linux MemAvailable, Windows ullAvailPhys), or 0 when it is unknown.
func AvailableGB() float64 { return availableGB() }
