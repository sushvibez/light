# Light Studio

Desktop AI studio (Wails v3 + React) that runs Light's models natively, in
process, on the local GPU. It is the same engine as `light serve` (`cmd/light`); the app
also exposes it as an OpenAI-compatible API on `http://127.0.0.1:8129/v1`
(override with `LIGHT_API_ADDR`).

## Run

```
cd studio
wails3 dev                        # hot reload (Vite + Go)
```

## Build

```
cd studio
wails3 generate bindings -ts -d frontend/bindings   # after changing service.go / engine types
(cd frontend && npm install && npm run build)
go build -o bin/studio .
./bin/studio
```

Requires the CUDA build prerequisites of the main module (or `-tags nocuda`
for CPU only), plus `libwebkit2gtk-4.1` and GTK 3 on Linux.

## Models

The sidebar lists what `engine.Catalog()` finds: supported `model_type`s in
the Hugging Face cache (llama, mistral, qwen2, phi3) and GGUF files under
`~/.cache/light/gguf` (or `LIGHT_GGUF_DIR`). A GGUF file needs a tokenizer:
either `<name>.tokenizer.json` next to it or a cached HF repo of the same
family and vocabulary size.

## Layout

- `main.go` — Wails application, window, API server goroutine
- `service.go` — `StudioService` bound to the frontend (Models, Status, Load, Unload, Chat, Stop); tokens stream as `token` / `done` events
- `frontend/src/App.tsx` — the UI; `frontend/bindings/` is generated
