// Example 42 — speech-to-text with Qwen3-ASR.
//
// Decodes a WAV file (any rate / channels; resampled to 16 kHz mono),
// computes the 128-bin log-mel spectrogram on the host, runs the AuT audio
// encoder once, splices the projected audio into the Qwen3 chat prompt and
// greedily decodes "language <Name><asr_text>…" with a KV cache.
//
//	go run ./examples/42_qwen3_asr -wav /usr/share/sounds/alsa/Front_Center.wav
//	go run ./examples/42_qwen3_asr -wav talk.wav -lang de -prompt "Hotwords: Light, Candle"
package main

import (
	"flag"
	"fmt"
	"log"
	"time"

	"light/engine"
	"light/models"
)

func main() {
	repo := flag.String("repo", "Qwen/Qwen3-ASR-1.7B-hf", "HF Qwen3-ASR repo")
	wav := flag.String("wav", "/usr/share/sounds/alsa/Front_Center.wav", "input WAV file")
	lang := flag.String("lang", "", "language code or name (empty = auto-detect)")
	prompt := flag.String("prompt", "", "system-prompt context / hotwords")
	devName := flag.String("device", "gpu", "gpu or cpu")
	dt := flag.String("dtype", "bf16", "decoder precision: bf16 or f32")
	flag.Parse()

	tr, err := engine.LoadTranscriber(engine.Spec{ID: *repo, Repo: *repo, Device: *devName, DType: *dt})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("loaded %s in %.1fs\n", tr.Describe(), tr.LoadTime.Seconds())
	t0 := time.Now()
	res, err := tr.TranscribeWAV(*wav, models.TranscribeOpts{Language: *lang, Prompt: *prompt})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("audio %.2fs | mel %.0f ms | encode %.0f ms | decode %.0f ms (%d tokens) | %.1fx real time | total %.0f ms\n",
		res.AudioSec, res.MelMS, res.EncodeMS, res.DecodeMS, res.Tokens, res.RealTimeX, float64(time.Since(t0).Milliseconds()))
	fmt.Printf("language: %s\ntext: %s\n", res.Language, res.Text)
}
