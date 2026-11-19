package nn

import (
	"fmt"

	"light/ops"
	"light/tensor"
)

// Merge returns one adapter equal to a + b: the ranks concatenate and each
// side's scale folds into its B rows, so several LoRAs on one layer cost
// one side path.
func (a *LoRA) Merge(b *LoRA) (*LoRA, error) {
	if a.AT.Dim(0) != b.AT.Dim(0) || a.BT.Dim(1) != b.BT.Dim(1) {
		return nil, fmt.Errorf("nn.LoRA.Merge: adapters (%d → %d) and (%d → %d)", a.AT.Dim(0), a.BT.Dim(1), b.AT.Dim(0), b.BT.Dim(1))
	}
	at, err := ops.Concat([]*tensor.Tensor{a.AT, b.AT}, 1)
	if err != nil {
		return nil, err
	}
	ba, err := ops.Affine(a.BT, a.Scale, 0)
	if err != nil {
		return nil, err
	}
	bb, err := ops.Affine(b.BT, b.Scale, 0)
	if err != nil {
		return nil, err
	}
	bt, err := ops.Concat([]*tensor.Tensor{ba, bb}, 0)
	if err != nil {
		return nil, err
	}
	return &LoRA{AT: at, BT: bt, Scale: 1}, nil
}

// AddLoRA installs ad on l, merging with an adapter already there.
func (l *Linear) AddLoRA(ad *LoRA) error {
	if l.LoRA == nil {
		l.LoRA = ad
		return nil
	}
	m, err := l.LoRA.Merge(ad)
	if err != nil {
		return err
	}
	l.LoRA = m
	return nil
}
