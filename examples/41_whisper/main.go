// Example 41 — speech-to-text with Whisper.
//
// Decodes a WAV file (any rate / channels; resampled to 16 kHz mono),
// computes Whisper's log-mel spectrogram on the host, runs the encoder once
// and greedily decodes the transcript with a KV cache.
//
//	go run ./examples/41_whisper -wav /usr/share/sounds/alsa/Front_Center.wav
//	go run ./examples/41_whisper -repo openai/whisper-base -wav talk.wav -lang de
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"light/audio"
	"light/device"
	"light/hub"
	"light/models"
	"light/tensor"
	"light/tokenizer"
)

func main() {
	repo := flag.String("repo", "openai/whisper-tiny", "HF Whisper repo")
	wav := flag.String("wav", "/usr/share/sounds/alsa/Front_Center.wav", "input WAV file")
	lang := flag.String("lang", "", "language code (empty = auto-detect)")
	translate := flag.Bool("translate", false, "translate to English")
	devName := flag.String("device", "gpu", "gpu or cpu")
	flag.Parse()

	dev := device.CPU()
	if *devName == "gpu" && device.CUDAAvailable() {
		d, err := device.CUDA(0)
		if err != nil {
			log.Fatal(err)
		}
		dev = d
	}

	t0 := time.Now()
	samples, rate, err := audio.ReadWAV(*wav)
	if err != nil {
		log.Fatal(err)
	}
	s16 := audio.Resample(samples, rate, audio.WhisperSampleRate)
	filters, err := models.LoadWhisperMelFilters(*repo)
	if err != nil {
		log.Fatal(err)
	}
	mel, err := audio.LogMel(s16, filters)
	if err != nil {
		log.Fatal(err)
	}
	tMel := time.Since(t0)

	t1 := time.Now()
	m, err := models.LoadWhisperFromHF(*repo, dev)
	if err != nil {
		log.Fatal(err)
	}
	tokPath, err := hub.Download(*repo, "main", "tokenizer.json")
	if err != nil {
		log.Fatal(err)
	}
	f, err := os.Open(tokPath)
	if err != nil {
		log.Fatal(err)
	}
	bpe, err := tokenizer.LoadBPE(f)
	f.Close()
	if err != nil {
		log.Fatal(err)
	}
	tLoad := time.Since(t1)

	melT, err := tensor.New(mel, []int{1, len(filters), audio.WhisperFrames}, dev)
	if err != nil {
		log.Fatal(err)
	}
	t2 := time.Now()
	enc, err := m.Encode(melT)
	if err != nil {
		log.Fatal(err)
	}
	tEnc := time.Since(t2)
	t3 := time.Now()
	ids, err := m.TranscribeEncoded(enc, models.TranscribeOpts{Language: *lang, Translate: *translate})
	if err != nil {
		log.Fatal(err)
	}
	tDec := time.Since(t3)
	ids32 := make([]int32, len(ids))
	for i, v := range ids {
		ids32[i] = int32(v)
	}
	fmt.Printf("device %v  model %s (%.1fM params)\n", dev, *repo, float64(m.ParamCount())/1e6)
	fmt.Printf("audio  %.2fs @ %d Hz → mel %.0f ms | load %v | encode %v | decode %d tok %v\n",
		float64(len(samples))/float64(rate), rate, float64(tMel.Microseconds())/1000, tLoad.Round(time.Millisecond), tEnc.Round(time.Millisecond), len(ids), tDec.Round(time.Millisecond))
	fmt.Printf("text:  %q\n", bpe.Decode(ids32))
}
