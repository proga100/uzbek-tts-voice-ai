"""Build chatterbox_feruza_lora.ipynb — Phase B: FeruzaSpeech -> Chatterbox LoRA.

Research/non-commercial proof (FeruzaSpeech academic + Chatterbox base). Edit cells
here, run `python build_finetune.py`, then push with the Kaggle CLI.
"""
import json, os

CELLS = []
def md(t): CELLS.append({"cell_type":"markdown","metadata":{},"source":t.strip("\n").splitlines(keepends=True)})
def code(t): CELLS.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":t.strip("\n").splitlines(keepends=True)})

md("""
# FeruzaSpeech -> Chatterbox v3 LoRA fine-tune (Phase B)

Teaches Chatterbox Uzbek by LoRA fine-tuning on the clean single-speaker FeruzaSpeech
corpus, using the `gokhaneraslan/chatterbox-finetuning` toolkit.

**Research / non-commercial only** (FeruzaSpeech academic license). Requires: GPU on,
and an `HF_TOKEN` Kaggle secret (you must accept FeruzaSpeech terms on HF first).
""")

md("## 1. Install (Chatterbox first, restore numpy, ffmpeg, toolkit)")
code("""
# Proven Chatterbox dep combo on Kaggle (from the smoke run):
#   chatterbox-tts -> toolkit reqs -> torchvision==0.21.0 -> numpy>=2.0 (LAST).
!pip install -q chatterbox-tts
import os
os.chdir("/kaggle/working")
if not os.path.exists("chatterbox-finetuning"):
    !git clone -q https://github.com/gokhaneraslan/chatterbox-finetuning.git
%cd /kaggle/working/chatterbox-finetuning
!pip install -q -r requirements.txt
!pip install -q torchvision==0.21.0          # match torch 2.6.0 (fix torchvision::nms)
!pip install -q -U "numpy>=2.0,<2.3"         # MUST be last: restore numpy 2.x ABI for torch
print("install done")
""")

md("## 2. Authenticate to Hugging Face (gated FeruzaSpeech)")
code("""
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
login(token=HF_TOKEN)
print("HF login OK")
""")

md("""
## 3. Build the dataset (FeruzaSpeech -> LJSpeech layout)

Toolkit expects `MyTTSDataset/wavs/*.wav` + `metadata.csv` rows
`name|raw_text|normalized_text`. We use the **Latin** transcript and keep clips
3-10 s. Start with a ~3 h subset for a faster first run.
""")
code("""
import os, csv, soundfile as sf
from datasets import load_dataset, Audio

OUT = "/kaggle/working/chatterbox-finetuning/MyTTSDataset"
os.makedirs(f"{OUT}/wavs", exist_ok=True)

raw = load_dataset("k2speech/FeruzaSpeech", token=HF_TOKEN)
split = "train" if "train" in raw else list(raw.keys())[0]
ds = raw[split]
print("columns:", ds.column_names)

# ADJUST after first run if names differ (print(ds.column_names) above).
TEXT_COL = next((c for c in ["text_latin","latin","transcription_latin","text"]
                 if c in ds.column_names), ds.column_names[-1])
ds = ds.cast_column("audio", Audio(sampling_rate=16000))

TARGET_SECONDS = 3 * 3600   # ~3 h subset
rows, total = [], 0.0
for i, ex in enumerate(ds):
    a = ex["audio"]; dur = len(a["array"]) / a["sampling_rate"]
    if dur < 3 or dur > 10:
        continue
    text = (ex[TEXT_COL] or "").strip()
    if not text:
        continue
    name = f"f{i:06d}"
    sf.write(f"{OUT}/wavs/{name}.wav", a["array"], a["sampling_rate"])
    rows.append((name, text, text))
    total += dur
    if total >= TARGET_SECONDS:
        break

with open(f"{OUT}/metadata.csv", "w", newline="") as f:
    w = csv.writer(f, delimiter="|")
    for r in rows:
        w.writerow(r)
print(f"wrote {len(rows)} clips, ~{total/3600:.2f} h")
print("sample:", rows[0])
""")

md("""
## 4. Download base models + read the vocab size

`setup.py` downloads the Chatterbox base + tokenizer and prints the vocab size we
must set in the config. Uzbek note: the default tokenizer covers Turkish characters;
if `oʻ gʻ ʻ ʼ` come out mispronounced, a custom tokenizer is the follow-up (ADJUST).
""")
code("""
%cd /kaggle/working/chatterbox-finetuning
!python setup.py 2>&1 | tee /kaggle/working/setup.log | tail -20
# Extract the vocab size the script reports.
import re
txt = open("/kaggle/working/setup.log").read()
m = re.findall(r"vocab[_ ]?size[^0-9]*(\\d+)", txt, flags=re.I)
VOCAB = int(m[-1]) if m else None
print("detected vocab size:", VOCAB, "(ADJUST manually if None)")
""")

md("## 5. Patch `src/config.py` for our dataset + LoRA on a 16 GB T4")
code("""
import re
cfg_path = "/kaggle/working/chatterbox-finetuning/src/config.py"
cfg = open(cfg_path).read()

def setv(text, name, value):
    # replace `name: type = <old>` keeping the annotation
    return re.sub(rf"({name}\\s*:\\s*[^=]+=\\s*)([^\\n#]+)", rf"\\g<1>{value}", text, count=1)

cfg = setv(cfg, "csv_path", '"./MyTTSDataset/metadata.csv"')
cfg = setv(cfg, "wav_dir", '"./MyTTSDataset/wavs"')
cfg = setv(cfg, "is_lora", "True")
cfg = setv(cfg, "is_turbo", "True")
cfg = setv(cfg, "ljspeech", "True")
cfg = setv(cfg, "preprocess", "True")
cfg = setv(cfg, "batch_size", "4")       # 32 default OOMs a T4
cfg = setv(cfg, "num_epochs", "10")
cfg = setv(cfg, "learning_rate", "1e-4")
if VOCAB:
    cfg = setv(cfg, "new_vocab_size", str(VOCAB))

open(cfg_path, "w").write(cfg)
print("patched config. Relevant lines:")
for line in cfg.splitlines():
    if any(k in line for k in ["csv_path","wav_dir","is_lora","is_turbo","batch_size",
                                "num_epochs","learning_rate","new_vocab_size","preprocess"]):
        print("  ", line.strip())
""")

md("## 6. Preprocess + train (LoRA)")
code("""
%cd /kaggle/working/chatterbox-finetuning
!python train.py
""")

md("""
## 7. Inference — Uzbek agriculture sentences in the fine-tuned voice

Uses a FeruzaSpeech clip as the speaker reference, then synthesizes domain sentences
(numbers, %, agro terms, the hard letters `oʻ gʻ q ng`).
""")
code("""
import os, glob, shutil
ref_src = sorted(glob.glob("/kaggle/working/chatterbox-finetuning/MyTTSDataset/wavs/*.wav"))[0]
os.makedirs("/kaggle/working/chatterbox-finetuning/speaker_reference", exist_ok=True)
shutil.copy(ref_src, "/kaggle/working/chatterbox-finetuning/speaker_reference/reference.wav")

SENTENCES = [
    "Salom! Men ovozli yordamchisiman.",
    "Pomidor bargida sariq dogʻlar paydo boʻldi.",
    "Fungitsidni 25 foiz konsentratsiyada, bir gektarga ikki litr soling.",
    "Bugʻdoy maydonida qoʻngʻir zang kasalligi koʻrindi.",
]
# ADJUST: confirm inference.py CLI/vars after first run; here we edit TEXT/AUDIO and run.
import re
inf = "/kaggle/working/chatterbox-finetuning/inference.py"
src = open(inf).read()
src = re.sub(r'TEXT_TO_SAY\\s*=\\s*.*', f'TEXT_TO_SAY = {SENTENCES[1]!r}', src, count=1)
src = re.sub(r'AUDIO_PROMPT\\s*=\\s*.*', 'AUDIO_PROMPT = "./speaker_reference/reference.wav"', src, count=1)
open(inf, "w").write(src)

%cd /kaggle/working/chatterbox-finetuning
!python inference.py
import shutil, os
if os.path.exists("output.wav"):
    shutil.copy("output.wav", "/kaggle/working/uz_finetuned.wav")
    print("saved /kaggle/working/uz_finetuned.wav")
from IPython.display import Audio, display
if os.path.exists("/kaggle/working/uz_finetuned.wav"):
    display(Audio("/kaggle/working/uz_finetuned.wav"))
""")

md("""
## Next steps
- Listen to `uz_finetuned.wav`. Judge Uzbek pronunciation + the agro terms/numbers.
- Raise the subset (step 3) + epochs (step 5) for quality.
- **Not for commercial production** (research license). Production = own recorded narrator
  + Chatterbox (MIT base) + commercial data.
""")

nb = {"cells":CELLS,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
      "language_info":{"name":"python"},"accelerator":"GPU"},"nbformat":4,"nbformat_minor":5}
out = os.path.join(os.path.dirname(__file__), "chatterbox_feruza_lora.ipynb")
json.dump(nb, open(out,"w"), indent=1)
print("wrote", out, len(CELLS), "cells")
