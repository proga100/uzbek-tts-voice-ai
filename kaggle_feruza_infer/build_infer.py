"""Build chatterbox_feruza_infer.ipynb — INFERENCE-ONLY sweep on the v11-trained voice.

No retraining. Mounts the v11 kernel output (trained LoRA adapter + Turbo base) and the
FeruzaSpeech dataset (for fresh reference clips), then sweeps reference-clip x temperature
on Uzbek test sentences. Goal: find the best synthesis settings before
spending GPU-hours on a bigger training run.

Inputs (attach in kernel-metadata.json):
  kernel_sources: ["rustamakhmedov/chatterbox-feruzaspeech-uzbek-lora"]   (trained model)
  dataset_sources: ["rustamakhmedov/feruzaspeech-uzbek-tts"]              (reference clips)
"""
import json, os

CELLS = []
def md(t): CELLS.append({"cell_type":"markdown","metadata":{},"source":t.strip("\n").splitlines(keepends=True)})
def code(t): CELLS.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":t.strip("\n").splitlines(keepends=True)})

md("""
# FeruzaSpeech voice — inference-only quality sweep (no retrain)

Loads the v11-trained LoRA adapter + Turbo base from the attached kernel output, and
sweeps **reference clip x temperature** on Uzbek test sentences. Listen to the grid
and pick the best settings.
""")

md("## 1. Install (same proven Chatterbox dep combo)")
code("""
import os
!apt-get -qq update && apt-get -qq install -y ffmpeg >/dev/null 2>&1
!pip install -q chatterbox-tts
!pip install -q torchvision==0.21.0
!pip install -q -U "numpy>=2.0,<2.3"
print("install done")
""")

md("""
## 2. Stage the trained model into /kaggle/working

The kernel output mounts read-only under /kaggle/input/.../chatterbox-finetuning. We copy
the parts inference needs (src, pretrained_models, chatterbox_output/new_lang_adapter,
*.py) into a writable working dir. We SKIP the big preprocess/ + wavs/ to save time/disk.
""")
code("""
import os, glob, shutil

# locate the mounted v11 output (folder containing chatterbox_output/new_lang_adapter)
SRC_BASE = None
for root, dirs, files in os.walk("/kaggle/input"):
    if root.endswith("chatterbox-finetuning") and os.path.isdir(os.path.join(root,"chatterbox_output")):
        SRC_BASE = root; break
if not SRC_BASE:
    # fallback: find adapter then walk up
    for root,_,files in os.walk("/kaggle/input"):
        if "adapter_model.safetensors" in files:
            SRC_BASE = root.split("/chatterbox_output")[0]; break
print("model source:", SRC_BASE)
assert SRC_BASE, "v11 output not found — attach kernel rustamakhmedov/chatterbox-feruzaspeech-uzbek-lora"

DST = "/kaggle/working/chatterbox-finetuning"
os.makedirs(DST, exist_ok=True)
# copy code + base models + adapter; skip preprocess cache and training wavs
for item in ["src","pretrained_models","setup.py","inference.py","merge_lora.py","train.py","requirements.txt"]:
    s = os.path.join(SRC_BASE, item)
    if os.path.exists(s):
        d = os.path.join(DST, item)
        if os.path.isdir(s): shutil.copytree(s, d, dirs_exist_ok=True)
        else: shutil.copy(s, d)
# adapter only (not the multi-GB checkpoint-4770)
os.makedirs(f"{DST}/chatterbox_output", exist_ok=True)
shutil.copytree(f"{SRC_BASE}/chatterbox_output/new_lang_adapter",
                f"{DST}/chatterbox_output/new_lang_adapter", dirs_exist_ok=True)
print("staged. pretrained:", os.listdir(f"{DST}/pretrained_models"))
print("adapter:", os.listdir(f"{DST}/chatterbox_output/new_lang_adapter"))
""")

md("""
## 3. Pick fresh reference clips from FeruzaSpeech

The reference clip drives timbre/prosody. v11 used just the single longest training clip.
Here we grab a few clean ~6-9 s clips straight from the dataset to compare.
""")
code("""
import os, csv, glob, shutil, soundfile as sf, librosa

# find FeruzaSpeech train.tsv in the attached dataset
TSV=None
for root,_,files in os.walk("/kaggle/input"):
    if "train.tsv" in files: TSV=os.path.join(root,"train.tsv"); FBASE=root; break
print("dataset:", TSV)

refs_dir="/kaggle/working/refs"; os.makedirs(refs_dir, exist_ok=True)
ref_paths=[]
if TSV:
    # pick a few clips with clean mid-length duration, spaced across the corpus
    rows=[]
    with open(TSV, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\\t"):
            try: d=float(row["duration"])
            except: continue
            if 6.0<=d<=9.0: rows.append(row["audio"])
    picks = [rows[i] for i in (10, len(rows)//2, len(rows)-10)] if len(rows)>20 else rows[:3]
    for i,rel in enumerate(picks):
        src=os.path.join(FBASE, rel)
        if not os.path.exists(src):
            hits=glob.glob(os.path.join(FBASE,"**",os.path.basename(rel)),recursive=True)
            src=hits[0] if hits else None
        if not src: continue
        y,sr=librosa.load(src, sr=24000, mono=True)
        p=f"{refs_dir}/ref{i+1}.wav"; sf.write(p,y,sr); ref_paths.append(p)
# always include the v11 reference for comparison if present
v11ref=glob.glob("/kaggle/input/**/speaker_reference/reference.wav", recursive=True)
if v11ref:
    shutil.copy(v11ref[0], f"{refs_dir}/ref_v11.wav"); ref_paths.append(f"{refs_dir}/ref_v11.wav")
print("reference clips:", ref_paths)
""")

md("## 4. Load the trained engine ONCE (Turbo + LoRA)")
code("""
import sys, os, torch
os.chdir("/kaggle/working/chatterbox-finetuning")
sys.path.insert(0, "/kaggle/working/chatterbox-finetuning")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import inference as inf            # imports cfg=TrainConfig(); does NOT run __main__
engine = inf.load_finetuned_engine_lora(DEVICE)
print("engine loaded on", DEVICE)
""")

md("""
## 5. Sweep reference x temperature on test sentences

Generates a labelled grid of wavs. Listen and tell me which (ref, temp) sounds best.
""")
code("""
import soundfile as sf, traceback

SENTENCES = {
  "greet": "Assalomu alaykum, sogʻ-salomatmisiz? Bugun qanday yordam bera olaman?",
  "city":  "Toshkent koʻchalarida yangi avtobuslar qatnovni boshladi.",
  "price": "Chiptalar narxi 25 foizga tushdi, eng arzoni 10 ming soʻm.",
}
OUT="/kaggle/working/sweep"; os.makedirs(OUT, exist_ok=True)
SR = engine.sr

def to_np(x):
    import numpy as np, torch
    # engine.generate / generate_sentence_audio may return tensor, ndarray, or (wav, sr)
    if isinstance(x, (tuple, list)):
        # pick the array-like element (the waveform), ignore the int sample-rate
        cand = [e for e in x if torch.is_tensor(e) or (hasattr(e, "__len__") and not isinstance(e, (str, bytes)))]
        x = cand[0] if cand else x[0]
    if torch.is_tensor(x): x = x.detach().cpu().numpy()
    return np.asarray(x, dtype="float32").squeeze()

def synth(text, ref, **kw):
    wav = engine.generate(text=text, audio_prompt_path=ref, **kw)   # call engine directly
    return to_np(wav)

results=[]
# (a) reference comparison: greeting sentence, temp 0.6, each reference clip
for ref in ref_paths:
    tag=os.path.splitext(os.path.basename(ref))[0]
    try:
        w=synth(SENTENCES["greet"], ref, temperature=0.6, repetition_penalty=1.2)
        p=f"{OUT}/greet_{tag}_t06.wav"; sf.write(p,w,SR); results.append(p); print("ok",p)
    except Exception: traceback.print_exc()
# (b) temperature comparison: greeting, best-guess first reference, temps
best_ref = ref_paths[0] if ref_paths else None
for t in [0.4, 0.6, 0.8]:
    try:
        w=synth(SENTENCES["greet"], best_ref, temperature=t, repetition_penalty=1.2)
        p=f"{OUT}/greet_ref1_t{int(t*10):02d}.wav"; sf.write(p,w,SR); results.append(p); print("ok",p)
    except Exception: traceback.print_exc()
# (c) the 3 test sentences with ref1 @ temp 0.6
for k,txt in SENTENCES.items():
    try:
        w=synth(txt, best_ref, temperature=0.6, repetition_penalty=1.2)
        p=f"{OUT}/sent_{k}_ref1_t06.wav"; sf.write(p,w,SR); results.append(p); print("ok",p)
    except Exception: traceback.print_exc()

print(f"\\nGENERATED {len(results)} clips in {OUT}")
""")

md("## 6. Listen")
code("""
from IPython.display import Audio, display
import os, glob
for p in sorted(glob.glob("/kaggle/working/sweep/*.wav")):
    print(os.path.basename(p)); display(Audio(p))
""")

nb={"cells":CELLS,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python"},"accelerator":"GPU"},"nbformat":4,"nbformat_minor":5}
out=os.path.join(os.path.dirname(__file__),"chatterbox_feruza_infer.ipynb")
json.dump(nb,open(out,"w"),indent=1)
print("wrote",out,len(CELLS),"cells")
