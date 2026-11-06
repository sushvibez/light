#!/usr/bin/env python3
"""Fixtures for npy/ and pth/ loader tests. Run from the repo root:
    .venv/bin/python scripts/verify/gen_npy_pth_fixtures.py
"""
import collections, numpy as np, torch

# ---- npy / npz --------------------------------------------------------------
a = np.arange(24, dtype=np.float32).reshape(2, 3, 4) * 0.5
np.save("npy/testdata/a_f32.npy", a)
np.save("npy/testdata/a_f32_fortran.npy", np.asfortranarray(a))
np.save("npy/testdata/scalar_f64.npy", np.float64(3.25))
np.savez("npy/testdata/mixed.npz",
         f16=a.astype(np.float16), f32=a, f64=a.astype(np.float64),
         i8=np.array([-3, 4], dtype=np.int8), u8=np.array([250, 7], dtype=np.uint8),
         i16=np.array([-300, 400], dtype=np.int16), u16=np.array([60000, 7], dtype=np.uint16),
         i32=np.array([-70000, 5], dtype=np.int32), u32=np.array([4000000000, 1], dtype=np.uint32),
         i64=np.array([-2**40, 9], dtype=np.int64), u64=np.array([2**63, 2], dtype=np.uint64),
         b=np.array([True, False, True]))
np.savez_compressed("npy/testdata/compressed.npz", x=a, y=a.T)

# ---- pth ---------------------------------------------------------------------
w = torch.arange(12, dtype=torch.float32).reshape(3, 4) * 0.25
sd = collections.OrderedDict()
sd["lin.weight"] = torch.nn.Parameter(w)
sd["lin.bias"] = torch.tensor([1.0, -2.0, 3.0])
sd["w_t"] = w.t()                       # non-contiguous view (shares storage)
sd["w_slice"] = w[1:, 1:3]              # offset + strides
sd["bf16"] = w.to(torch.bfloat16)
sd["f16"] = w.to(torch.float16)
sd["f64"] = w.double()
sd["i64"] = torch.tensor([[-5, 6], [7, 8]])
sd["i32"] = torch.tensor([9, -10], dtype=torch.int32)
sd["i16"] = torch.tensor([11, -12], dtype=torch.int16)
sd["i8"] = torch.tensor([13, -14], dtype=torch.int8)
sd["u8"] = torch.tensor([255, 16], dtype=torch.uint8)
sd["bool"] = torch.tensor([True, False])
sd["scalar"] = torch.tensor(2.5)
torch.save(sd, "pth/testdata/state_dict.pth")                   # protocol 2 (default)
torch.save(sd, "pth/testdata/state_dict_p4.pth", pickle_protocol=4)
torch.save({"epoch": 3, "state_dict": sd, "lr": 0.1}, "pth/testdata/checkpoint.pth")
m = torch.nn.Linear(4, 3); m.weight.data.copy_(w); m.bias.data.copy_(sd["lin.bias"])
torch.save(m, "pth/testdata/module.pth")                        # whole nn.Module pickled
print("ok")
