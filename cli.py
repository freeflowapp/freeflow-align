"""Run the aligner locally without Cog: python cli.py AUDIO TRANSCRIPT_FILE [--mode both]."""

from __future__ import annotations

import argparse
import json
import time

import align
from align import backends

p = argparse.ArgumentParser()
p.add_argument("audio")
p.add_argument("transcript", help="file holding plain text or the JSON segments format")
p.add_argument("--aligner", default=backends.DEFAULT, choices=sorted(backends.BACKENDS))
p.add_argument("--mode", default="segments", choices=align.MODES)
p.add_argument("--padding", type=float, default=1.5)
p.add_argument("--device", default=None)
p.add_argument("--out", default="-")
p.add_argument("--separate", action="store_true")
a = p.parse_args()

t0 = time.time()
with open(a.transcript, encoding="utf-8") as f:
    result = align.align_file(a.audio, f.read(), a.aligner, a.mode, a.padding, a.device, separate=a.separate)
result["seconds_taken"] = round(time.time() - t0, 1)
text = json.dumps(result, indent=1)
if a.out == "-":
    print(text)
else:
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(text)
