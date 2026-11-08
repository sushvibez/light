package main

import (
	"errors"
	"flag"
	"log"
	"net/http"

	"light/engine"
)

// cmdServe exposes the engine through an OpenAI-compatible HTTP API
// (/v1/chat/completions, /v1/audio/transcriptions, /v1/embeddings,
// /v1/audio/conversations; models load on request or with -model).
//
//	light serve -model qwen2.5-0.5b
//	curl localhost:8129/v1/chat/completions -d '{"messages":[{"role":"user","content":"hi"}],"stream":true}'
func cmdServe(args []string) error {
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	addr := fs.String("addr", "127.0.0.1:8129", "listen address")
	model := fs.String("model", "", "model to load at start (see `light list`)")
	var mf modelFlags
	mf.register(fs)
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() > 0 {
		return errors.New("usage: light serve [-addr host:port] [-model <model>] [-device gpu|cpu] [-dtype f32|bf16]")
	}
	rt := engine.NewRuntime()
	if *model != "" {
		entries, err := catalog()
		if err != nil {
			return err
		}
		e, err := resolve(entries, *model, nil)
		if err != nil {
			return err
		}
		spec := mf.spec(e)
		switch e.Kind {
		case "stt":
			_, err = rt.LoadSTT(spec)
		case "embedding":
			_, err = rt.LoadEmbedder(spec)
		case "speech":
			_, err = rt.LoadSpeech(spec)
		default:
			_, err = rt.Load(spec)
		}
		if err != nil {
			return err
		}
		log.Printf("loaded %s", rt.Describe())
	}
	log.Printf("listening on http://%s/v1", *addr)
	return http.ListenAndServe(*addr, engine.OpenAIHandler(rt))
}
