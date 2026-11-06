// Light Studio: a desktop chat app (Wails v3 + React) that runs Light's
// models natively in-process on the local GPU, and exposes the same engine
// as an OpenAI-compatible API on http://127.0.0.1:8129/v1 for other clients.
//
//	cd studio && wails3 dev        # hot-reload development
//	cd studio && wails3 build      # bin/studio
package main

import (
	"embed"
	"log"
	"net/http"
	"os"

	"github.com/wailsapp/wails/v3/pkg/application"

	"light/engine"
)

//go:embed all:frontend/dist
var assets embed.FS

func init() {
	application.RegisterEvent[Token]("token")
	application.RegisterEvent[Done]("done")
}

func main() {
	rt := engine.NewRuntime()
	svc := NewStudioService(rt)

	addr := os.Getenv("LIGHT_API_ADDR")
	if addr == "" {
		addr = "127.0.0.1:8129"
	}
	go func() {
		log.Printf("OpenAI-compatible API on http://%s/v1", addr)
		if err := http.ListenAndServe(addr, engine.OpenAIHandler(rt)); err != nil {
			log.Printf("api server: %v", err)
		}
	}()

	app := application.New(application.Options{
		Name:        "Light Studio",
		Description: "Local AI studio running Light models natively",
		Services:    []application.Service{application.NewService(svc)},
		Assets:      application.AssetOptions{Handler: application.AssetFileServerFS(assets)},
		Mac:         application.MacOptions{ApplicationShouldTerminateAfterLastWindowClosed: true},
	})
	svc.app = app
	svc.apiAddr = addr

	app.Window.NewWithOptions(application.WebviewWindowOptions{
		Title:            "Light Studio",
		Width:            1240,
		Height:           800,
		MinWidth:         720,
		MinHeight:        480,
		BackgroundColour: application.NewRGB(14, 15, 19),
		URL:              "/",
	})
	if err := app.Run(); err != nil {
		log.Fatal(err)
	}
}
