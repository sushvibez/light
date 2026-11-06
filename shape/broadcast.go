package shape

import "fmt"

// Broadcast returns the shape resulting from broadcasting a and b together
// using NumPy / PyTorch rules: align shapes to the right, each axis must be
// equal or one side must be 1. Returns an error on incompatible shapes.
func Broadcast(a, b Shape) (Shape, error) {
	ra, rb := a.Rank(), b.Rank()
	r := max(ra, rb)
	out := make([]int, r)
	for i := range r {
		// i counts from the left in `out`; align from the right.
		axA := ra - r + i
		axB := rb - r + i
		var da, db int
		if axA < 0 {
			da = 1
		} else {
			da = a.Dim(axA)
		}
		if axB < 0 {
			db = 1
		} else {
			db = b.Dim(axB)
		}
		switch {
		case da == db:
			out[i] = da
		case da == 1:
			out[i] = db
		case db == 1:
			out[i] = da
		default:
			return Shape{}, fmt.Errorf("shape: cannot broadcast %v and %v (axis %d: %d vs %d)", a, b, i, da, db)
		}
	}
	return Shape{dims: out}, nil
}

// BroadcastN broadcasts a list of shapes together, left-to-right.
func BroadcastN(shapes ...Shape) (Shape, error) {
	if len(shapes) == 0 {
		return Shape{}, fmt.Errorf("shape: BroadcastN requires at least one shape")
	}
	cur := shapes[0]
	for i := 1; i < len(shapes); i++ {
		var err error
		cur, err = Broadcast(cur, shapes[i])
		if err != nil {
			return Shape{}, err
		}
	}
	return cur, nil
}

// BroadcastStrides adapts a tensor with shape `from` and strides `fromStrides`
// to target shape `to` by setting broadcast-axis strides to 0. The returned
// Strides has rank equal to `to.Rank()`.
// If `from` cannot broadcast to `to`, returns an error.
func BroadcastStrides(from Shape, fromStrides Strides, to Shape) (Strides, error) {
	rFrom, rTo := from.Rank(), to.Rank()
	if rFrom > rTo {
		return Strides{}, fmt.Errorf("shape: cannot broadcast rank %d to rank %d", rFrom, rTo)
	}
	out := make([]int, rTo)
	for i := range rTo {
		axFrom := rFrom - rTo + i
		if axFrom < 0 {
			// Prepended axis — stride 0.
			out[i] = 0
			continue
		}
		dFrom := from.Dim(axFrom)
		dTo := to.Dim(i)
		switch dFrom {
		case dTo:
			out[i] = fromStrides.strides[axFrom]
		case 1:
			out[i] = 0
		default:
			return Strides{}, fmt.Errorf("shape: cannot broadcast %v to %v (axis %d: %d vs %d)", from, to, i, dFrom, dTo)
		}
	}
	return Strides{strides: out}, nil
}
