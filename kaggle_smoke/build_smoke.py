"""Build chatterbox_smoke.ipynb — full smoke + zero-shot EN/UZ with the dep fixes."""
import json, os

CELLS = []
def md(t): CELLS.append({"cell_type":"markdown","metadata":{},"source":t.strip("\n").splitlines(keepends=True)})
def code(t): CELLS.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":t.strip("\n").splitlines(keepends=True)})

md("""
# Chatterbox v3 — smoke + zero-shot Uzbek (Phase A)

Dependency fixes baked in:
- `chatterbox-tts` pins torch==2.6.0 but leaves torchvision mismatched →
  `torchvision::nms does not exist`. Fix: install **torchvision==0.21.0**.
- numpy 2.2.6 is fine at runtime (chatterbox's `numpy<2` pin is just a pip warning).
""")

md("## 1. Install (with torchvision fix), before importing torch")
code("""
!pip install -q chatterbox-tts
!pip install -q torchvision==0.21.0          # match torch 2.6.0 (fix torchvision::nms)
!pip install -q -U "numpy>=2.0,<2.3"         # MUST be last: restore numpy 2.x ABI for torch
print("install done")
""")

md("## 2. Versions + import check")
code("""
import importlib.metadata as m
for p in ["torch","torchvision","torchaudio","transformers","numpy"]:
    try: print(p, m.version(p))
    except Exception as e: print(p, "?", e)

import traceback
try:
    from transformers import LlamaModel  # the import that was failing
    print("transformers Llama import: OK")
except Exception:
    traceback.print_exc()
""")

md("## 3. Load the multilingual model")
code("""
import torch, traceback
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
model = None
try:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    print("loaded ChatterboxMultilingualTTS")
except Exception:
    traceback.print_exc()
    from chatterbox.tts import ChatterboxTTS
    model = ChatterboxTTS.from_pretrained(device=device)
    print("loaded ChatterboxTTS (base)")
""")

md("## 4. Zero-shot English (sanity)")
code("""
import torchaudio, traceback
try:
    try: wav = model.generate("Hello, this is a quick Chatterbox test.", language_id="en")
    except TypeError: wav = model.generate("Hello, this is a quick Chatterbox test.")
    torchaudio.save("/kaggle/working/en.wav", wav.cpu(), model.sr)
    print("saved en.wav @", model.sr)
except Exception:
    traceback.print_exc()
""")

md("## 5. Zero-shot Uzbek (the experiment) — try uz, tr, and no tag")
code("""
import torchaudio, traceback
UZ = "Salom! Bugun sizga qanday yordam bera olaman?"
for tag in ["uz","tr",None]:
    try:
        wav = model.generate(UZ) if tag is None else model.generate(UZ, language_id=tag)
        name = "uz_notag" if tag is None else f"uz_{tag}"
        torchaudio.save(f"/kaggle/working/{name}.wav", wav.cpu(), model.sr)
        print("OK ->", name + ".wav")
    except Exception as e:
        print(f"tag={tag} failed:", repr(e)[:160])
""")

md("## 6. Listen")
code("""
from IPython.display import Audio, display
import os
for f in ["en.wav","uz_uz.wav","uz_tr.wav","uz_notag.wav"]:
    p=f"/kaggle/working/{f}"
    if os.path.exists(p):
        print(f); display(Audio(p))
""")

nb = {"cells":CELLS,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
      "language_info":{"name":"python"},"accelerator":"GPU"},"nbformat":4,"nbformat_minor":5}
out = os.path.join(os.path.dirname(__file__), "chatterbox_smoke.ipynb")
json.dump(nb, open(out,"w"), indent=1)
print("wrote", out, len(CELLS), "cells")
