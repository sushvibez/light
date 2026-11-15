// Package splat renders 3D Gaussian splats on the CPU: EWA projection of
// each Gaussian to a screen-space ellipse, depth sort, front-to-back alpha
// compositing (the 3DGS forward pass without tiling). Meant for previews
// of generated objects, not real-time viewing.
package splat

import (
	"image"
	"image/color"
	"math"
	"runtime"
	"sort"
	"sync"
)

// Gaussians are activated splats: positions, degree-0 SH colours, scales,
// unit rotations (w, x, y, z) and opacities, flattened.
type Gaussians struct {
	XYZ, FeaturesDC, Scaling, Rotation, Opacity []float32
}

// Camera looks at the origin from a point on a sphere, z up.
type Camera struct {
	Yaw, Pitch float64 // radians: position = r·(sin yaw cos pitch, cos yaw cos pitch, sin pitch)
	Radius     float64
	FOV        float64 // vertical, degrees
}

const shC0 = 0.28209479177387814

type proj struct {
	x, y, depth     float64
	a, b, c         float64 // inverse 2-D covariance (conic)
	r, g, bl, alpha float64
	x0, y0, x1, y1  int
}

// Render draws g from cam into a w × h image over bg.
func Render(g *Gaussians, cam Camera, w, h int, bg color.NRGBA) *image.NRGBA {
	n := len(g.Opacity)
	cy, sy := math.Cos(cam.Yaw), math.Sin(cam.Yaw)
	cp, sp := math.Cos(cam.Pitch), math.Sin(cam.Pitch)
	pos := [3]float64{cam.Radius * sy * cp, cam.Radius * cy * cp, cam.Radius * sp}
	// View axes: forward to the origin, right, up.
	fwd := [3]float64{-pos[0] / cam.Radius, -pos[1] / cam.Radius, -pos[2] / cam.Radius}
	right := norm(cross(fwd, [3]float64{0, 0, 1}))
	if right == [3]float64{} {
		right = [3]float64{1, 0, 0}
	}
	up := cross(right, fwd)
	focal := float64(h) / 2 / math.Tan(cam.FOV*math.Pi/360)
	ps := make([]proj, 0, n)
	for i := range n {
		p := [3]float64{float64(g.XYZ[3*i]) - pos[0], float64(g.XYZ[3*i+1]) - pos[1], float64(g.XYZ[3*i+2]) - pos[2]}
		vx, vy, vz := dot(p, right), dot(p, up), dot(p, fwd)
		if vz < 0.01 {
			continue
		}
		// 3-D covariance R S Sᵀ Rᵀ.
		qw, qx, qy, qz := float64(g.Rotation[4*i]), float64(g.Rotation[4*i+1]), float64(g.Rotation[4*i+2]), float64(g.Rotation[4*i+3])
		R := [3][3]float64{
			{1 - 2*(qy*qy+qz*qz), 2 * (qx*qy - qw*qz), 2 * (qx*qz + qw*qy)},
			{2 * (qx*qy + qw*qz), 1 - 2*(qx*qx+qz*qz), 2 * (qy*qz - qw*qx)},
			{2 * (qx*qz - qw*qy), 2 * (qy*qz + qw*qx), 1 - 2*(qx*qx+qy*qy)},
		}
		s := [3]float64{float64(g.Scaling[3*i]), float64(g.Scaling[3*i+1]), float64(g.Scaling[3*i+2])}
		var M [3][3]float64 // R·S
		for r := range 3 {
			for c := range 3 {
				M[r][c] = R[r][c] * s[c]
			}
		}
		var S3 [3][3]float64
		for r := range 3 {
			for c := range 3 {
				S3[r][c] = M[r][0]*M[c][0] + M[r][1]*M[c][1] + M[r][2]*M[c][2]
			}
		}
		// World → view rotation W (rows: right, up, fwd), then the projection Jacobian J.
		W := [3][3]float64{right, up, fwd}
		J := [2][3]float64{{focal / vz, 0, -focal * vx / (vz * vz)}, {0, focal / vz, -focal * vy / (vz * vz)}}
		var T [2][3]float64 // J·W
		for r := range 2 {
			for c := range 3 {
				T[r][c] = J[r][0]*W[0][c] + J[r][1]*W[1][c] + J[r][2]*W[2][c]
			}
		}
		var cov [2][2]float64
		for r := range 2 {
			for c := range 2 {
				for k := range 3 {
					for l := range 3 {
						cov[r][c] += T[r][k] * S3[k][l] * T[c][l]
					}
				}
			}
		}
		cov[0][0] += 0.3 // the 3DGS low-pass
		cov[1][1] += 0.3
		det := cov[0][0]*cov[1][1] - cov[0][1]*cov[1][0]
		if det <= 0 {
			continue
		}
		mid := (cov[0][0] + cov[1][1]) / 2
		lmax := mid + math.Sqrt(math.Max(0.1, mid*mid-det))
		rad := math.Ceil(3 * math.Sqrt(lmax))
		sx, sy := float64(w)/2+focal*vx/vz, float64(h)/2-focal*vy/vz
		x0, x1 := int(math.Max(0, sx-rad)), int(math.Min(float64(w-1), sx+rad))
		y0, y1 := int(math.Max(0, sy-rad)), int(math.Min(float64(h-1), sy+rad))
		if x0 > x1 || y0 > y1 {
			continue
		}
		ps = append(ps, proj{x: sx, y: sy, depth: vz,
			a: cov[1][1] / det, b: -cov[0][1] / det, c: cov[0][0] / det,
			r:     clamp01(0.5 + shC0*float64(g.FeaturesDC[3*i])),
			g:     clamp01(0.5 + shC0*float64(g.FeaturesDC[3*i+1])),
			bl:    clamp01(0.5 + shC0*float64(g.FeaturesDC[3*i+2])),
			alpha: float64(g.Opacity[i]), x0: x0, y0: y0, x1: x1, y1: y1})
	}
	sort.Slice(ps, func(i, j int) bool { return ps[i].depth < ps[j].depth })
	// Composite in row bands, each band walking the sorted splats.
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	bands := runtime.GOMAXPROCS(0) * 4
	var wg sync.WaitGroup
	for band := range bands {
		wg.Add(1)
		go func(band int) {
			defer wg.Done()
			ya, yb := band*h/bands, (band+1)*h/bands
			acc := make([][4]float64, (yb-ya)*w) // r, g, b, transmittance
			for i := range acc {
				acc[i][3] = 1
			}
			for _, p := range ps {
				if p.y1 < ya || p.y0 >= yb {
					continue
				}
				for y := max(p.y0, ya); y <= min(p.y1, yb-1); y++ {
					dy := float64(y) + 0.5 - p.y
					for x := p.x0; x <= p.x1; x++ {
						px := &acc[(y-ya)*w+x]
						if px[3] < 1e-4 {
							continue
						}
						dx := float64(x) + 0.5 - p.x
						e := -0.5 * (p.a*dx*dx + 2*p.b*dx*dy + p.c*dy*dy)
						if e > 0 {
							continue
						}
						al := math.Min(0.99, p.alpha*math.Exp(e))
						if al < 1.0/255 {
							continue
						}
						wgt := al * px[3]
						px[0] += wgt * p.r
						px[1] += wgt * p.g
						px[2] += wgt * p.bl
						px[3] *= 1 - al
					}
				}
			}
			for y := ya; y < yb; y++ {
				for x := range w {
					a := acc[(y-ya)*w+x]
					t := a[3]
					o := 4 * (y*w + x)
					img.Pix[o] = q8(a[0] + t*float64(bg.R)/255)
					img.Pix[o+1] = q8(a[1] + t*float64(bg.G)/255)
					img.Pix[o+2] = q8(a[2] + t*float64(bg.B)/255)
					img.Pix[o+3] = 255
				}
			}
		}(band)
	}
	wg.Wait()
	return img
}

// Turntable renders `views` yaws around cam (same pitch, radius, FOV) side
// by side, each w × h.
func Turntable(g *Gaussians, cam Camera, views, w, h int, bg color.NRGBA) *image.NRGBA {
	out := image.NewNRGBA(image.Rect(0, 0, views*w, h))
	for v := range views {
		c := cam
		c.Yaw += 2 * math.Pi * float64(v) / float64(views)
		im := Render(g, c, w, h, bg)
		for y := range h {
			copy(out.Pix[(y*views*w+v*w)*4:(y*views*w+v*w+w)*4], im.Pix[y*w*4:(y+1)*w*4])
		}
	}
	return out
}

func dot(a, b [3]float64) float64 { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] }
func cross(a, b [3]float64) [3]float64 {
	return [3]float64{a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]}
}
func norm(a [3]float64) [3]float64 {
	l := math.Sqrt(dot(a, a))
	if l == 0 {
		return [3]float64{}
	}
	return [3]float64{a[0] / l, a[1] / l, a[2] / l}
}
func clamp01(v float64) float64 { return math.Min(math.Max(v, 0), 1) }
func q8(v float64) uint8        { return uint8(math.Round(clamp01(v) * 255)) }
