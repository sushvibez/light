#!/usr/bin/env python3
"""torchvision.ops.deform_conv2d references (CPU, F32) for ops/deform_conv_test.go:
k = 1, 3, 7 with BiRefNet's same padding, offsets large enough to leave the
image, a mask in [0, 2] (BiRefNet's 2·sigmoid), bias or none; and a strided,
dilated case.

    python scripts/verify/ref_deform_conv.py <out.json>
"""
import json, sys

import torch
from torchvision.ops import deform_conv2d

torch.manual_seed(0)
r = lambda t: [round(v, 7) for v in t.flatten().tolist()]
cases = []
for k, stride, dil, bias in ((1, 1, 1, True), (3, 1, 1, False), (7, 1, 1, True), (3, 2, 2, True)):
    B, C, H, W, O = 2, 3, 9, 11, 4
    pad = dil * (k // 2)
    Ho = (H + 2 * pad - dil * (k - 1) - 1) // stride + 1
    Wo = (W + 2 * pad - dil * (k - 1) - 1) // stride + 1
    x = torch.randn(B, C, H, W)
    off = torch.randn(B, 2 * k * k, Ho, Wo) * 2.5
    mask = 2 * torch.sigmoid(torch.randn(B, k * k, Ho, Wo))
    w = torch.randn(O, C, k, k) * 0.3
    b = torch.randn(O) if bias else None
    y = deform_conv2d(x, off, w, b, stride=(stride, stride), padding=(pad, pad), dilation=(dil, dil), mask=mask)
    cases.append({"k": k, "stride": stride, "dilation": dil, "padding": pad, "dims": [B, C, H, W, O, Ho, Wo],
                  "x": r(x), "offset": r(off), "mask": r(mask), "weight": r(w), "bias": r(b) if bias else None,
                  "out": r(y)})
json.dump(cases, open(sys.argv[1], "w"))
print(len(cases), "cases")
