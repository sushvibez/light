// Example 30 — YOLOv5n forward pass on 640×640 input.
//
//	go run            ./examples/30_yolov5n
//	go run            ./examples/30_yolov5n -size s -nc 80
package main

import (
	"flag"
	"fmt"
	"log"
	"runtime"
	"time"

	"light/device"
	"light/dtype"
	"light/models"
	"light/tensor"
)

func main() {
	deviceFlag := flag.String("device", "gpu", "gpu or cpu")
	sizeFlag := flag.String("size", "n", "n | s | m | l | x")
	nc := flag.Int("nc", 80, "number of classes (COCO=80)")
	imgsz := flag.Int("imgsz", 640, "input image size (square)")
	flag.Parse()

	dev := pickDevice(*deviceFlag)
	fmt.Printf("device: %v\n", dev)

	var cfg models.YOLOv5Config
	switch *sizeFlag {
	case "n":
		cfg = models.YOLOv5nConfig()
	case "s":
		cfg = models.YOLOv5sConfig()
	case "m":
		cfg = models.YOLOv5mConfig()
	case "l":
		cfg = models.YOLOv5lConfig()
	case "x":
		cfg = models.YOLOv5xConfig()
	default:
		log.Fatalf("unknown size %q", *sizeFlag)
	}
	cfg.NumClasses = *nc

	t0 := time.Now()
	y5, err := models.NewYOLOv5(cfg, dev)
	if err != nil {
		log.Fatal(err)
	}
	y5.SetTraining(false)
	fmt.Printf("YOLOv5%s  %.2fM params (constructed in %v)\n", *sizeFlag,
		float64(y5.ParamCount())/1e6, time.Since(t0).Round(time.Millisecond))

	x, err := tensor.Rand(dtype.F32, []int{1, 3, *imgsz, *imgsz}, dev)
	if err != nil {
		log.Fatal(err)
	}
	t1 := time.Now()
	outs, err := y5.Forward(x)
	if err != nil {
		log.Fatalf("forward: %v", err)
	}
	runtime.GC()
	fmt.Printf("input  %v   forward %v\n", x.Shape(), time.Since(t1).Round(time.Millisecond))
	for i, o := range outs {
		fmt.Printf("  head[%d] (stride %d): %v\n", i, y5.Strides[i], o.Shape())
	}
}

func pickDevice(name string) device.Device {
	if name == "gpu" && device.CUDAAvailable() {
		runtime.LockOSThread()
		g, err := device.CUDA(0)
		if err == nil {
			return g
		}
		fmt.Printf("CUDA unavailable (%v); falling back to CPU\n", err)
	}
	return device.CPU()
}
