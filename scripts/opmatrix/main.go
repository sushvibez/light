// Command opmatrix prints the op × device × dtype coverage matrix from the
// live kernel registry. It is the source for the "Kernel coverage" section of
// FEATURES.md: run it after adding kernels and paste the table.
//
//	go run ./scripts/opmatrix              # CPU + CUDA (default build)
//	go run -tags webgpu ./scripts/opmatrix # adds the WebGPU column
//	go run ./scripts/opmatrix -md          # markdown table
package main

import (
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"

	"light/device"
	"light/dtype"
	"light/internal/registry"
	_ "light/ops" // links kernels/cpu, kernels/cuda (and kernels/webgpu with -tags webgpu)
)

var kinds = []device.Kind{device.KindCPU, device.KindCUDA, device.KindWebGPU}

func main() {
	md := flag.Bool("md", false, "emit a markdown table")
	flag.Parse()

	byOp := map[string]map[device.Kind][]string{}
	for _, k := range registry.Registered() {
		if byOp[k.Op] == nil {
			byOp[k.Op] = map[device.Kind][]string{}
		}
		byOp[k.Op][k.Dev] = append(byOp[k.Op][k.Dev], k.DType.String())
	}
	ops := make([]string, 0, len(byOp))
	for op := range byOp {
		ops = append(ops, op)
	}
	sort.Strings(ops)
	for _, op := range ops {
		for _, dts := range byOp[op] {
			sort.Slice(dts, func(i, j int) bool { return dtypeRank(dts[i]) < dtypeRank(dts[j]) })
		}
	}

	w := os.Stdout
	if *md {
		fmt.Fprintln(w, "| Op | CPU | CUDA | WebGPU |")
		fmt.Fprintln(w, "|----|-----|------|--------|")
	}
	cpuOnly := 0
	for _, op := range ops {
		row := byOp[op]
		if len(row[device.KindCUDA]) == 0 {
			cpuOnly++
		}
		if *md {
			fmt.Fprintf(w, "| `%s` | %s | %s | %s |\n", op,
				cell(row[device.KindCPU]), cell(row[device.KindCUDA]), cell(row[device.KindWebGPU]))
			continue
		}
		fmt.Fprintf(w, "%-24s cpu=%-28s cuda=%-20s webgpu=%s\n", op,
			strings.Join(row[device.KindCPU], ","), strings.Join(row[device.KindCUDA], ","),
			strings.Join(row[device.KindWebGPU], ","))
	}
	fmt.Fprintf(os.Stderr, "%d ops, %d without a CUDA kernel\n", len(ops), cpuOnly)
}

func cell(dts []string) string {
	if len(dts) == 0 {
		return "—"
	}
	return strings.Join(dts, " ")
}

func dtypeRank(s string) int {
	for i, d := range []dtype.DType{dtype.F16, dtype.BF16, dtype.F32, dtype.F64, dtype.U8, dtype.U32, dtype.I64} {
		if d.String() == s {
			return i
		}
	}
	return 100
}
