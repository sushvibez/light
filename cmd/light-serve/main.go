// light-serve exposes any catalog model through an OpenAI-compatible HTTP
// API, so existing chat clients and SDKs can talk to Light directly.
//
//	go run ./cmd/light-serve -model gguf/qwen2-0.5b-instruct-q4_k_m.gguf
//	curl localhost:8129/v1/chat/completions -d '{"messages":[{"role":"user","content":"hi"}],"stream":true}'
package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"

	"light/engine"
)

func main() {
	addr := flag.String("addr", "127.0.0.1:8129", "listen address")
	model := flag.String("model", "", "catalog id to load at start (see -list)")
	dev := flag.String("device", "gpu", "gpu or cpu")
	dt := flag.String("dtype", "", "f32 or bf16 for safetensors weights (default: catalog choice)")
	list := flag.Bool("list", false, "print the catalog and exit")
	flag.Parse()

	entries, err := engine.Catalog()
	if err != nil {
		log.Fatal(err)
	}
	if *list || *model == "" {
		fmt.Println("models:")
		for _, e := range entries {
			fmt.Printf("  %-55s %-12s %-7s %6.1f GB\n", e.Spec.ID, e.Kind, e.Params, float64(e.SizeBytes)/(1<<30))
		}
		if *list {
			return
		}
	}
	rt := engine.NewRuntime()
	if *model != "" {
		var spec *engine.Spec
		for _, e := range entries {
			if e.Spec.ID == *model {
				s := e.Spec
				spec = &s
			}
		}
		if spec == nil {
			fmt.Fprintf(os.Stderr, "model %q not in catalog\n", *model)
			os.Exit(1)
		}
		spec.Device = *dev
		if *dt != "" {
			spec.DType = *dt
		}
		m, err := rt.Load(*spec)
		if err != nil {
			log.Fatal(err)
		}
		log.Printf("loaded %s (%s family, %s chat template) in %v", m.Spec.ID, m.Spec.Family, m.Template.Source, m.LoadTime)
	}
	log.Printf("listening on http://%s/v1", *addr)
	log.Fatal(http.ListenAndServe(*addr, engine.OpenAIHandler(rt)))
}
