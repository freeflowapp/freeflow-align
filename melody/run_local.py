"""Run the melody pipeline on one file without Cog or Docker (CPU/MPS/CUDA) and write the JSON the model returns.

Usage: MELODY_CKPT=/path/to/ckpt python run_local.py <mix audio> <out.json>
"""
import sys
import time

import pitch

t0 = time.time()
sep = pitch.Separator()
print("device", sep.device, f"model load {time.time() - t0:.1f} s", flush=True)
out = pitch.run(sep, sys.argv[1])
open(sys.argv[2], "w").write(out)
print(sys.argv[2], f"{len(out) / 1024:.0f} KB, total {time.time() - t0:.1f} s", flush=True)
