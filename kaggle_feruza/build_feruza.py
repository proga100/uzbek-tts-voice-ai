"""Build chatterbox_feruza_lora.ipynb — Chatterbox LoRA on FeruzaSpeech (clean studio voice).

Same proven pipeline as the Common Voice run, but with FeruzaSpeech: ONE clean
professional female speaker (no mining needed). Requires an `HF_TOKEN` Kaggle secret
(FeruzaSpeech is gated). Research/non-commercial (academic license) — proof of quality.
"""
import json, os

CELLS = []
def md(t): CELLS.append({"cell_type":"markdown","metadata":{},"source":t.strip("\n").splitlines(keepends=True)})
def code(t): CELLS.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":t.strip("\n").splitlines(keepends=True)})

md("""
# FeruzaSpeech -> Chatterbox LoRA (clean studio Uzbek voice)

One clean professional female narrator, ~90 min subset, 30 epochs. Same fixed pipeline
as the Common Voice run; the only change is the dataset (clean studio vs consumer-mic).
Requires an `HF_TOKEN` Kaggle secret (gated dataset). Research/non-commercial.
""")

md("## 1. Install (proven Chatterbox dep combo) + toolkit + ffmpeg")
code("""
import os
# Avoid HF Xet storage endpoint (it 429-rate-limited our gated download); use std LFS.
!apt-get -qq update && apt-get -qq install -y ffmpeg >/dev/null 2>&1
!pip install -q chatterbox-tts hf_xet
os.chdir("/kaggle/working")
if not os.path.exists("chatterbox-finetuning"):
    !git clone -q https://github.com/gokhaneraslan/chatterbox-finetuning.git
%cd /kaggle/working/chatterbox-finetuning
!pip install -q -r requirements.txt
!pip install -q torchvision==0.21.0
!pip install -q -U "numpy>=2.0,<2.3"
print("install done")
""")

md("## 2. Authenticate to Hugging Face (gated FeruzaSpeech)")
code("""
import time
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
HF_TOKEN = None
for attempt in range(6):
    try:
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
        break
    except Exception as e:
        print(f"secret read attempt {attempt+1} failed ({type(e).__name__}); retry in 20s")
        time.sleep(20)
if not HF_TOKEN:
    raise RuntimeError("Could not read HF_TOKEN secret after retries")
login(token=HF_TOKEN)
print("HF login OK")
""")

md("""
## 3. Build dataset (FeruzaSpeech -> LJSpeech, ~90 min)

Single speaker, so no mining. Latin transcript, 3-10 s clips. Decode with
soundfile/librosa (avoids torchcodec).
""")
code("""
import os, csv, io, tempfile, time
import soundfile as sf, librosa, numpy as np
from datasets import load_dataset, Audio

OUT = "/kaggle/working/chatterbox-finetuning/MyTTSDataset"
os.makedirs(f"{OUT}/wavs", exist_ok=True)

# ADJUST dataset id if you were approved for the 44.1k TTS variant:
#   k2speech/FeruzaSpeech_44100_Hz_tts
# Retry with backoff in case HF still 429-throttles.
raw = None
for attempt in range(4):
    try:
        raw = load_dataset("k2speech/FeruzaSpeech", token=HF_TOKEN)
        break
    except Exception as e:
        wait = 120 * (attempt + 1)
        print(f"load attempt {attempt+1} failed ({type(e).__name__}); retry in {wait}s")
        time.sleep(wait)
if raw is None:
    raise RuntimeError("FeruzaSpeech download failed after retries (HF rate limit?)")
split = "train" if "train" in raw else list(raw.keys())[0]
ds = raw[split]
print("split:", split, "| columns:", ds.column_names)

# Prefer a Latin-script transcript column.
TEXT_COL = next((c for c in ["text_latin","latin","transcription_latin","sentence","text","transcription"]
                 if c in ds.column_names), ds.column_names[-1])
print("using text column:", TEXT_COL)
AUDIO_COL = "audio" if "audio" in ds.column_names else next(c for c in ds.column_names if "audio" in c.lower())
sub = ds.cast_column(AUDIO_COL, Audio(decode=False))

def load_audio(a):
    if isinstance(a, dict) and a.get("bytes"):
        try: y, sr0 = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
        except Exception:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(a["bytes"]); p = f.name
            y, sr0 = librosa.load(p, sr=None, mono=True)
    else:
        path = a["path"] if isinstance(a, dict) else a
        y, sr0 = librosa.load(path, sr=None, mono=True)
    if getattr(y, "ndim", 1) > 1: y = y.mean(axis=1)
    if sr0 != 16000:
        y = librosa.resample(y.astype("float32"), orig_sr=sr0, target_sr=16000); sr0 = 16000
    return y.astype("float32"), sr0

rows, total = [], 0.0
for i, ex in enumerate(sub):
    try: y, sr0 = load_audio(ex[AUDIO_COL])
    except Exception: continue
    dur = len(y) / sr0
    if dur < 3 or dur > 10: continue
    text = (ex[TEXT_COL] or "").strip()
    if not text: continue
    name = f"fz{i:06d}"
    sf.write(f"{OUT}/wavs/{name}.wav", y, sr0)
    rows.append((name, text, text)); total += dur
    if total >= 90*60: break

with open(f"{OUT}/metadata.csv","w",newline="") as f:
    w=csv.writer(f,delimiter="|")
    for r in rows: w.writerow(r)
print(f"wrote {len(rows)} clips, ~{total/60:.1f} min")
print("sample:", rows[0] if rows else "NONE")
""")

md("## 4. Patch src/config.py BEFORE setup (turbo base)")
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

md("## 5. Download base models (TURBO)")
code("""
%cd /kaggle/working/chatterbox-finetuning
import subprocess, sys
r = subprocess.run([sys.executable, "setup.py"])
if r.returncode != 0: raise RuntimeError("setup.py failed")
print("pretrained_models:", os.listdir("pretrained_models"))
""")

md("## 6. Preprocess + train (LoRA)")
code("""
%cd /kaggle/working/chatterbox-finetuning
import subprocess, sys
r = subprocess.run([sys.executable, "train.py"])
if r.returncode != 0: raise RuntimeError("train.py failed")
print("training done")
""")

md("## 7. Inference — Uzbek sentences in the clean fine-tuned voice")
code("""
import os, glob, shutil, re, subprocess, sys
ref = max(glob.glob("/kaggle/working/chatterbox-finetuning/MyTTSDataset/wavs/*.wav"), key=os.path.getsize)
os.makedirs("/kaggle/working/chatterbox-finetuning/speaker_reference", exist_ok=True)
shutil.copy(ref, "/kaggle/working/chatterbox-finetuning/speaker_reference/reference.wav")

TEXT = "Salom! Pomidor bargida sariq dogʻlar paydo boʻldi. Nima qilishim kerak?"
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
