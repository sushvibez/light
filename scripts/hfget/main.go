// hfget downloads one file of a Hugging Face repo into the shared HF cache
// through hub.Download (resumable, HF_TOKEN honoured) and prints its path.
//
//	go run ./scripts/hfget kyutai/moshiko-candle-q8 model.q8.gguf [revision]
package main

import (
	"fmt"
	"os"

	"light/hub"
)

func main() {
	if len(os.Args) < 3 {
		fmt.Fprintln(os.Stderr, "usage: hfget repo filename [revision]")
		os.Exit(2)
	}
	rev := ""
	if len(os.Args) > 3 {
		rev = os.Args[3]
	}
	p, err := hub.Download(os.Args[1], rev, os.Args[2])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println(p)
}
