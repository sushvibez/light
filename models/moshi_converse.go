package models

import (
	"fmt"

	"light/device"
	"light/ops"
	"light/tensor"
)

// MoshiTurn is the result of a conversation turn.
type MoshiTurn struct {
	Audio      *tensor.Tensor // Moshi's voice, (1, 1, L) F32 at 24 kHz
	TextTokens []int64        // text stream, pad (3) and end-of-pad (0) included
	Frames     int            // frames stepped (user frames + extra)
}

// Words returns the text tokens with Moshi's pad / end-of-pad markers
// removed, ready for the SentencePiece decoder.
func (t *MoshiTurn) Words() []int32 {
	out := make([]int32, 0, len(t.TextTokens))
	for _, id := range t.TextTokens {
		if id != 0 && id != 3 {
			out = append(out, int32(id))
		}
	}
	return out
}

// Converse runs one full-duplex turn frame by frame, the way the live
// system does: each 80 ms of the user's 24 kHz mono audio (1, 1, L; zero
// padded to whole frames) is encoded by the streaming Mimi, stepped through
// the protocol, and Moshi's own codes for the frame that just completed
// are decoded straight back to 80 ms of audio. extraFrames more steps with
// a silent (pad) user give the model room to finish its reply. Nil
// samplers mean argmax.
func (m *Moshi) Converse(mimi *Mimi, userAudio *tensor.Tensor, extraFrames int, textSampler, audioSampler func([]float32) int) (*MoshiTurn, error) {
	cfg := m.Cfg
	if userAudio.Rank() != 3 || userAudio.Dim(0) != 1 {
		return nil, fmt.Errorf("moshi: Converse takes one utterance (1, 1, L), got %v", userAudio.Shape())
	}
	userCB := cfg.AudioCodebooks - cfg.DepQ
	hop := mimi.Cfg.HopLength()
	L := userAudio.Dim(2)
	T := (L + hop - 1) / hop
	audio := userAudio
	if pad := T*hop - L; pad > 0 {
		var err error
		if audio, err = ops.Pad(userAudio, 2, 0, pad); err != nil {
			return nil, err
		}
	}
	dev := userAudio.Device()
	codec := mimi.NewStream()
	g := NewMoshiGen(m, textSampler, audioSampler)
	prev := cfg.TextStartToken()
	var out []*tensor.Tensor
	for step := 0; step < T+extraFrames; step++ {
		var user []int64
		if step < T {
			chunk, err := audio.Narrow(2, step*hop, hop)
			if err != nil {
				return nil, err
			}
			if chunk, err = chunk.Contiguous(); err != nil {
				return nil, err
			}
			codes, err := codec.EncodeStep(chunk, userCB) // (1, userCB, 1)
			if err != nil {
				return nil, err
			}
			host, err := codes.To(device.CPU())
			if err != nil {
				return nil, err
			}
			user = append([]int64(nil), tensor.MustSlice[int64](host)...)
		}
		var err error
		if prev, err = g.Step(prev, user); err != nil {
			return nil, err
		}
		own := g.LastAudioTokens()
		if own == nil {
			continue
		}
		ownT, err := tensor.New(own, []int{1, cfg.DepQ, 1}, dev)
		if err != nil {
			return nil, err
		}
		pcm, err := codec.DecodeStep(ownT)
		if err != nil {
			return nil, err
		}
		out = append(out, pcm)
	}
	turn := &MoshiTurn{TextTokens: g.TextTokens(), Frames: T + extraFrames}
	if len(out) > 0 {
		var err error
		if turn.Audio, err = ops.Concat(out, 2); err != nil {
			return nil, err
		}
	}
	return turn, nil
}
