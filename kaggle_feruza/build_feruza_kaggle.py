"""Build chatterbox_feruza_lora.ipynb — FeruzaSpeech (clean studio Uzbek) -> Chatterbox LoRA.

This variant reads FeruzaSpeech from an ATTACHED KAGGLE DATASET (a full clone of
`k2speech/FeruzaSpeech`), NOT from Hugging Face. So: no HF token, no Kaggle secret,
no Xet 429 — the run is fully API-triggerable. Attach the dataset
`rustamakhmedov/feruzaspeech-uzbek-tts` to the kernel (kernel-metadata.json).
"""
import json, os

CELLS = []
def md(t): CELLS.append({"cell_type":"markdown","metadata":{},"source":t.strip("\n").splitlines(keepends=True)})
def code(t): CELLS.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":t.strip("\n").splitlines(keepends=True)})

md("""
# FeruzaSpeech -> Chatterbox LoRA (clean studio Uzbek voice)

One clean professional female narrator, ~90 min subset, 30 epochs. Same fixed pipeline
as the Common Voice run; the only change is the data source: a **Kaggle dataset clone**
of FeruzaSpeech (no Hugging Face token / secret / Xet at runtime). Research/non-commercial.
""")

md("## 1. Install (proven Chatterbox dep combo) + toolkit + ffmpeg")
code("""
import os
!apt-get -qq update && apt-get -qq install -y ffmpeg >/dev/null 2>&1
!pip install -q chatterbox-tts
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
## 2. Build dataset from the attached Kaggle clone (FeruzaSpeech -> LJSpeech, ~90 min)

Finds the FeruzaSpeech dir under `/kaggle/input/...` (looks for `train.tsv`), selects
3-10 s clips up to ~90 min using the Latin transcript, resamples to 16 kHz with
soundfile/librosa (avoids torchcodec), writes `MyTTSDataset/{wavs,metadata.csv}`.
""")
code("""
import os, csv, glob, soundfile as sf, librosa, numpy as np

# locate the attached dataset (the folder that contains train.tsv)
TSV = None
for root, _, files in os.walk("/kaggle/input"):
    if "train.tsv" in files:
        TSV = os.path.join(root, "train.tsv"); BASE = root; break
if not TSV:
    raise RuntimeError("train.tsv not found under /kaggle/input — attach the FeruzaSpeech dataset")
print("dataset base:", BASE)

OUT = "/kaggle/working/chatterbox-finetuning/MyTTSDataset"
os.makedirs(f"{OUT}/wavs", exist_ok=True)

# select 3-10 s clips up to ~90 min from the TSV (audio | text_latin | text_cyrillic | duration | words_count)
sel, total = [], 0.0
with open(TSV, encoding="utf-8") as f:
    r = csv.DictReader(f, delimiter="\\t")
    for row in r:
        try: dur = float(row["duration"])
        except Exception: continue
        if dur < 3 or dur > 10: continue
        txt = (row.get("text_latin") or "").strip()
        if not txt: continue
        sel.append((row["audio"], txt)); total += dur
        if total >= 90*60: break
print(f"selected {len(sel)} clips, ~{total/60:.1f} min")

rows = []
for i, (rel, txt) in enumerate(sel):
    src = os.path.join(BASE, rel)
    if not os.path.exists(src):  # some clones nest under train/ ; try a glob fallback
        hits = glob.glob(os.path.join(BASE, "**", os.path.basename(rel)), recursive=True)
        if not hits: continue
        src = hits[0]
    try:
        y, sr0 = sf.read(src, dtype="float32")
    except Exception:
        y, sr0 = librosa.load(src, sr=None, mono=True)
    if getattr(y, "ndim", 1) > 1: y = y.mean(axis=1)
    if sr0 != 16000:
        y = librosa.resample(y.astype("float32"), orig_sr=sr0, target_sr=16000); sr0 = 16000
    name = f"fz{i:05d}"
    sf.write(f"{OUT}/wavs/{name}.wav", y, sr0)
    rows.append((name, txt, txt))

with open(f"{OUT}/metadata.csv","w",newline="",encoding="utf-8") as f:
    w = csv.writer(f, delimiter="|")
    for r in rows: w.writerow(r)
print(f"wrote {len(rows)} clips")
print("sample:", rows[0] if rows else "NONE")
""")

md("## 3. Patch src/config.py BEFORE setup (turbo base)")
code("""
import re
p="/kaggle/working/chatterbox-finetuning/src/config.py"; cfg=open(p).read()
def setv(t,n,v): return re.sub(rf"({n}\\s*:\\s*[^=]+=\\s*)([^\\n#]+)", rf"\\g<1>{v}", t, count=1)
cfg=setv(cfg,"csv_path",'"./MyTTSDataset/metadata.csv"')
cfg=setv(cfg,"wav_dir",'"./MyTTSDataset/wavs"')
cfg=setv(cfg,"is_lora","True"); cfg=setv(cfg,"is_turbo","True")
cfg=setv(cfg,"ljspeech","True"); cfg=setv(cfg,"preprocess","True")
cfg=setv(cfg,"batch_size","4"); cfg=setv(cfg,"num_epochs","30"); cfg=setv(cfg,"learning_rate","1e-4")
# Disk guard: full checkpoints (model+optimizer, several GB each) every 500 steps x keep-5
# fills /kaggle/working (~20GB) by step ~2500. Keep only 1, save rarely -> ~15GB freed.
cfg=setv(cfg,"save_steps","2000"); cfg=setv(cfg,"save_total_limit","1")
open(p,"w").write(cfg)
for line in cfg.splitlines():
    if any(k in line for k in ["csv_path","wav_dir","is_lora","is_turbo","batch_size","num_epochs","learning_rate","new_vocab_size","preprocess","save_steps","save_total_limit"]):
        print(" ",line.strip())
""")

md("## 4. Download base models (TURBO)")
code("""
%cd /kaggle/working/chatterbox-finetuning
import subprocess, sys, os
r = subprocess.run([sys.executable, "setup.py"])
if r.returncode != 0: raise RuntimeError("setup.py failed")
print("pretrained_models:", os.listdir("pretrained_models"))
""")

md("## 5. Preprocess + train (LoRA)")
code("""
%cd /kaggle/working/chatterbox-finetuning
import subprocess, sys
r = subprocess.run([sys.executable, "train.py"])
if r.returncode != 0: raise RuntimeError("train.py failed")
print("training done")
""")

md("## 6. Inference — Uzbek sentence in the clean fine-tuned voice")
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
    shutil.copy("output.wav","/kaggle/working/feruza_uz_finetuned.wav")
    print("saved /kaggle/working/feruza_uz_finetuned.wav")
from IPython.display import Audio, display
if os.path.exists("/kaggle/working/feruza_uz_finetuned.wav"):
    display(Audio("/kaggle/working/feruza_uz_finetuned.wav"))
""")

md("""
## Result
`feruza_uz_finetuned.wav` — clean studio-trained Uzbek voice. Compare to the Common
Voice version. Research/non-commercial license.
""")

nb={"cells":CELLS,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python"},"accelerator":"GPU"},"nbformat":4,"nbformat_minor":5}
out=os.path.join(os.path.dirname(__file__),"chatterbox_feruza_lora.ipynb")
json.dump(nb,open(out,"w"),indent=1)
print("wrote",out,len(CELLS),"cells")
