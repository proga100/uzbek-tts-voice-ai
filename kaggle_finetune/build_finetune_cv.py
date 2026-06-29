"""Build chatterbox_cv_lora.ipynb — Phase B PROOF on Common Voice Uzbek (CC0, ungated).

Goal: prove the Chatterbox LoRA fine-tune pipeline runs end-to-end on Kaggle, using a
single mined speaker from Common Voice Uzbek. No HF token / no gating needed.

Key ordering: patch src/config.py (is_turbo=True) BEFORE setup.py, so setup downloads
the TURBO base model the train/inference steps expect. train/inference raise on failure
(so a broken step shows ERROR, not a false COMPLETE).
"""
import json, os

CELLS = []
def md(t): CELLS.append({"cell_type":"markdown","metadata":{},"source":t.strip("\n").splitlines(keepends=True)})
def code(t): CELLS.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":t.strip("\n").splitlines(keepends=True)})

md("""
# Common Voice Uzbek -> Chatterbox LoRA (pipeline PROOF)

Mines the most-prolific speaker from the open CC0 `yakhyo/mozilla-common-voice-uzbek`,
then LoRA-fine-tunes Chatterbox (Turbo/multilingual) on it. No HF token needed. The
voice is an arbitrary volunteer (consumer-mic) — this validates the training loop, not
final quality. Swap in FeruzaSpeech later for a clean voice.
""")

md("## 1. Install (proven Chatterbox dep combo) + toolkit + ffmpeg")
code("""
!apt-get -qq update && apt-get -qq install -y ffmpeg >/dev/null 2>&1
!pip install -q chatterbox-tts
import os
os.chdir("/kaggle/working")
if not os.path.exists("chatterbox-finetuning"):
    !git clone -q https://github.com/gokhaneraslan/chatterbox-finetuning.git
%cd /kaggle/working/chatterbox-finetuning
!pip install -q -r requirements.txt
!pip install -q torchvision==0.21.0          # match torch 2.6.0 (fix torchvision::nms)
!pip install -q -U "numpy>=2.0,<2.3"         # LAST: restore numpy 2.x ABI for torch
print("install done")
""")

md("""
## 2. Build dataset: mine the most-prolific clean speaker

Group Common Voice by `client_id`, pick the speaker with the most up-voted clips, keep
3-10 s clips, ~40 min total. Decode with soundfile/librosa (avoids torchcodec).
""")
code("""
import os, csv, io, tempfile
import soundfile as sf, librosa, numpy as np
from collections import Counter
from datasets import load_dataset, Audio

OUT = "/kaggle/working/chatterbox-finetuning/MyTTSDataset"
os.makedirs(f"{OUT}/wavs", exist_ok=True)

ds = load_dataset("yakhyo/mozilla-common-voice-uzbek", split="train")
print("rows:", len(ds), "| columns:", ds.column_names)
ids = ds["client_id"]; up = ds["up_votes"]; dn = ds["down_votes"]
# Stricter quality: clearly up-voted, zero down-votes.
def clean(u,d): return (u or 0) >= 2 and (d or 0) == 0
good = Counter(cid for cid,u,d in zip(ids,up,dn) if clean(u,d))
top_id, top_n = good.most_common(1)[0]
print(f"top speaker {top_id[:12]}... has {top_n} clean clips")

idx = [i for i,(cid,u,d) in enumerate(zip(ids,up,dn)) if cid==top_id and clean(u,d)]
sub = ds.select(idx).cast_column("audio", Audio(decode=False))
TEXT_COL = "text" if "text" in sub.column_names else "sentence"

def load_audio(a):
    if a.get("bytes"):
        try:
            y, sr0 = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
        except Exception:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                f.write(a["bytes"]); p = f.name
            y, sr0 = librosa.load(p, sr=None, mono=True)
    else:
        y, sr0 = librosa.load(a["path"], sr=None, mono=True)
    if getattr(y, "ndim", 1) > 1: y = y.mean(axis=1)
    if sr0 != 16000:
        y = librosa.resample(y.astype("float32"), orig_sr=sr0, target_sr=16000); sr0 = 16000
    return y.astype("float32"), sr0

rows, total = [], 0.0
for i, ex in enumerate(sub):
    try: y, sr0 = load_audio(ex["audio"])
    except Exception: continue
    dur = len(y) / sr0
    if dur < 3 or dur > 10: continue
    text = (ex[TEXT_COL] or "").strip()
    if not text: continue
    name = f"cv{i:06d}"
    sf.write(f"{OUT}/wavs/{name}.wav", y, sr0)
    rows.append((name, text, text)); total += dur
    if total >= 90*60: break

with open(f"{OUT}/metadata.csv","w",newline="") as f:
    w=csv.writer(f,delimiter="|")
    for r in rows: w.writerow(r)
print(f"wrote {len(rows)} clips, ~{total/60:.1f} min from one speaker")
print("sample:", rows[0] if rows else "NONE")
""")

md("""
## 3. Patch src/config.py BEFORE setup (so setup downloads the TURBO base)

is_turbo=True -> setup.py fetches t3_turbo_v1.safetensors (+ multilingual tokenizer),
which the train/inference steps expect. new_vocab_size keeps its default
`52260 if is_turbo else 2454`, so no manual vocab patching needed.
""")
code("""
import re
p="/kaggle/working/chatterbox-finetuning/src/config.py"; cfg=open(p).read()
def setv(t,n,v): return re.sub(rf"({n}\\s*:\\s*[^=]+=\\s*)([^\\n#]+)", rf"\\g<1>{v}", t, count=1)
cfg=setv(cfg,"csv_path",'"./MyTTSDataset/metadata.csv"')
cfg=setv(cfg,"wav_dir",'"./MyTTSDataset/wavs"')
cfg=setv(cfg,"is_lora","True"); cfg=setv(cfg,"is_turbo","True")
cfg=setv(cfg,"ljspeech","True"); cfg=setv(cfg,"preprocess","True")
cfg=setv(cfg,"batch_size","4"); cfg=setv(cfg,"num_epochs","30"); cfg=setv(cfg,"learning_rate","1e-4")
open(p,"w").write(cfg)
for line in cfg.splitlines():
    if any(k in line for k in ["csv_path","wav_dir","is_lora","is_turbo","batch_size","num_epochs","learning_rate","new_vocab_size","preprocess"]):
        print(" ",line.strip())
""")

md("## 4. Download base models (TURBO, per the patched config)")
code("""
%cd /kaggle/working/chatterbox-finetuning
import subprocess, sys
r = subprocess.run([sys.executable, "setup.py"])
if r.returncode != 0: raise RuntimeError("setup.py failed")
print("=== pretrained_models ==="); print(os.listdir("pretrained_models"))
""")

md("## 5. Preprocess + train (LoRA) — raises on failure")
code("""
%cd /kaggle/working/chatterbox-finetuning
import subprocess, sys
r = subprocess.run([sys.executable, "train.py"])
if r.returncode != 0: raise RuntimeError("train.py failed")
print("training done")
""")

md("## 6. Inference — Uzbek sentence in the fine-tuned voice")
code("""
import os, glob, shutil, re, subprocess, sys
ref = max(glob.glob("/kaggle/working/chatterbox-finetuning/MyTTSDataset/wavs/*.wav"), key=os.path.getsize)
os.makedirs("/kaggle/working/chatterbox-finetuning/speaker_reference", exist_ok=True)
shutil.copy(ref, "/kaggle/working/chatterbox-finetuning/speaker_reference/reference.wav")

TEXT = "Salom! Bugun sizga qanday yordam bera olaman?"
inf="/kaggle/working/chatterbox-finetuning/inference.py"; src=open(inf).read()
src=re.sub(r'TEXT_TO_SAY\\s*=\\s*.*', f'TEXT_TO_SAY = {TEXT!r}', src, count=1)
src=re.sub(r'AUDIO_PROMPT\\s*=\\s*.*', 'AUDIO_PROMPT = "./speaker_reference/reference.wav"', src, count=1)
open(inf,"w").write(src)

%cd /kaggle/working/chatterbox-finetuning
r = subprocess.run([sys.executable, "inference.py"])
if r.returncode != 0: raise RuntimeError("inference.py failed")
if os.path.exists("output.wav"):
    shutil.copy("output.wav","/kaggle/working/cv_uz_finetuned.wav")
    print("saved /kaggle/working/cv_uz_finetuned.wav")
from IPython.display import Audio, display
if os.path.exists("/kaggle/working/cv_uz_finetuned.wav"):
    display(Audio("/kaggle/working/cv_uz_finetuned.wav"))
""")

md("""
## Result
`cv_uz_finetuned.wav` proves the pipeline trains an Uzbek voice end-to-end on Kaggle.
Quality is limited (consumer-mic volunteer). Swap dataset to FeruzaSpeech for a clean
studio voice; same notebook otherwise.
""")

nb={"cells":CELLS,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python"},"accelerator":"GPU"},"nbformat":4,"nbformat_minor":5}
out=os.path.join(os.path.dirname(__file__),"chatterbox_cv_lora.ipynb")
json.dump(nb,open(out,"w"),indent=1)
print("wrote",out,len(CELLS),"cells")
