"""Build mms_uzbek_feruza_finetune.ipynb — fine-tune MMS Uzbek (Cyrillic) on FeruzaSpeech.

Base: facebook/mms-tts-uzb-script_cyrillic (a good Uzbek VITS the user approved).
Toolkit: ylacombe/finetune-hf-vits. Data: FeruzaSpeech Cyrillic transcripts (clean,
single studio speaker) from the attached Kaggle dataset rustamakhmedov/feruzaspeech-uzbek-tts.

No HF download of audio at train time (reads /kaggle/input); only pulls the base model +
discriminator (public) and pushes the prepared text/audio dataset to a private Hub repo so
the training script can load it.
"""
import json, os

CELLS = []
def md(t): CELLS.append({"cell_type":"markdown","metadata":{},"source":t.strip("\n").splitlines(keepends=True)})
def code(t): CELLS.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":t.strip("\n").splitlines(keepends=True)})

md("""
# MMS Uzbek (Cyrillic) -> FeruzaSpeech fine-tune (VITS)

Start from `facebook/mms-tts-uzb-script_cyrillic` (already speaks correct Uzbek) and adapt it
to the FeruzaSpeech studio voice. Single speaker, ~5 h clean Cyrillic clips. Cyrillic needs
no uroman. Toolkit: ylacombe/finetune-hf-vits.
""")

md("## 1. Install toolkit + build monotonic_align")
code("""
import os, torch
os.chdir("/kaggle/working")
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
      "| preinstalled torch:", torch.__version__)
# Kaggle's default torch (2.10+cu128) has no kernel image for this GPU. Install a known-good
# build (2.6.0 ran fine for Chatterbox; supports T4 sm_75 and P100 sm_60), pin it so the
# toolkit installs below can't swap it back.
!pip install -q torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 "matplotlib==3.9.2"
CON="/kaggle/working/constraints.txt"
open(CON,"w").write("torch==2.6.0\\ntorchvision==0.21.0\\ntorchaudio==2.6.0\\nmatplotlib==3.9.2\\n")
if not os.path.exists("finetune-hf-vits"):
    !git clone -q https://github.com/ylacombe/finetune-hf-vits.git
%cd /kaggle/working/finetune-hf-vits
!pip install -q -c $CON -r requirements.txt
# Toolkit needs transformers 4.x (5.0 dropped VitsConfig.pad_token_id) and datasets<4.
!pip install -q -c $CON "transformers==4.46.3" "datasets[audio]>=2.16,<3.0" "tokenizers<0.21"
print("install done (training subprocess will use torch 2.6.0)")
# Cython monotonic alignment search (required)
%cd /kaggle/working/finetune-hf-vits/monotonic_align
!mkdir -p monotonic_align
!python setup.py build_ext --inplace
%cd /kaggle/working/finetune-hf-vits
# Patch: load a LOCAL dataset dir (our HF token is read-only, can't push to Hub).
P="/kaggle/working/finetune-hf-vits/run_vits_finetuning.py"; s=open(P).read()
inject='''
from datasets import load_from_disk as _lfd
_ORIG_LD = load_dataset
def load_dataset(name, *a, **k):
    import os as _os
    if isinstance(name, str) and _os.path.isdir(name):
        d=_lfd(name); sp=k.get("split")
        return d[sp] if (sp and hasattr(d,"keys")) else d
    return _ORIG_LD(name, *a, **k)
'''
if "_ORIG_LD" not in s:
    s=s.replace("from datasets import DatasetDict, load_dataset",
                "from datasets import DatasetDict, load_dataset"+inject, 1)
# single-speaker: collator may omit speaker_id -> make the forward reads safe (None is fine)
s=s.replace('speaker_id=batch["speaker_id"]', 'speaker_id=batch.get("speaker_id")')
open(P,"w").write(s)
print("patched run_vits_finetuning for local dataset + speaker_id")
import transformers, datasets, accelerate
print("transformers", transformers.__version__, "| datasets", datasets.__version__, "| accelerate", accelerate.__version__)
print("install done")
""")

md("## 2. Hugging Face login (push prepared dataset + load base)")
code("""
import os
from huggingface_hub import login
# Provide your token via env var HF_TOKEN or a Kaggle secret named HF_TOKEN.
HF_TOKEN = os.environ.get("HF_TOKEN", "hf_PUT_YOUR_TOKEN_HERE")
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN") or HF_TOKEN
except Exception:
    pass
os.environ["HF_TOKEN"] = HF_TOKEN
login(token=HF_TOKEN)
print("HF login OK")
""")

md("## 3. Add a discriminator to the MMS Uzbek checkpoint (needed for VITS training)")
code("""
%cd /kaggle/working/finetune-hf-vits
import subprocess, sys, os
DISC_DIR = "/kaggle/working/mms_uzb_disc"
if not os.path.exists(os.path.join(DISC_DIR, "config.json")):
    r = subprocess.run([sys.executable, "convert_original_discriminator_checkpoint.py",
                        "--language_code", "uzb-script_cyrillic",
                        "--pytorch_dump_folder_path", DISC_DIR])
    if r.returncode != 0: raise RuntimeError("discriminator conversion failed")
print("disc model:", os.listdir(DISC_DIR))
""")

md("""
## 4. Build a clean Cyrillic dataset from FeruzaSpeech and push to the Hub

Keep clips 1.5-12 s whose Cyrillic transcript uses only MMS-vocab Uzbek letters (drops stray
Latin / rare chars / digit clips). Lowercase. Single speaker. Push to a private Hub repo so
the training script can load it by name.
""")
code("""
import os, csv, glob
from datasets import Dataset, Audio

ALLOWED = set("абвгдежзийклмнопрстуфхчшъьэюяёўғқҳ")
TSV=None
for root,_,files in os.walk("/kaggle/input"):
    if "train.tsv" in files: TSV=os.path.join(root,"train.tsv"); BASE=root; break
assert TSV, "FeruzaSpeech train.tsv not found under /kaggle/input"
print("dataset:", BASE)

# FeruzaSpeech is already 16 kHz mono — hand paths straight to `datasets` (no librosa).
src_paths, texts, total = [], [], 0.0
with open(TSV, encoding="utf-8") as f:
    for row in csv.DictReader(f, delimiter="\\t"):
        try: dur=float(row["duration"])
        except: continue
        if dur<1.5 or dur>12: continue
        txt=(row.get("text_cyrillic") or "").strip().lower()
        letters=set(c for c in txt if c.isalpha())
        if not letters or (letters-ALLOWED): continue
        rel=row["audio"]; src=os.path.join(BASE, rel)
        if not os.path.exists(src):
            hits=glob.glob(os.path.join(BASE,"**",os.path.basename(rel)),recursive=True)
            if not hits: continue
            src=hits[0]
        src_paths.append(src); texts.append(txt); total+=dur
print(f"clean clips: {len(src_paths)}  ~{total/3600:.2f} h")

from datasets import DatasetDict
ds = Dataset.from_dict({"audio": src_paths, "text": texts,
                        "speaker_id": [0]*len(src_paths)}).cast_column("audio", Audio(sampling_rate=16000))
DS_PATH="/kaggle/working/fz_ds"
DatasetDict({"train": ds}).save_to_disk(DS_PATH)
print("saved dataset ->", DS_PATH, "| clips:", len(ds))
""")

md("## 5. Write the training config")
code("""
import json
cfg = {
  "project_name": "mms_uzb_feruza",
  "push_to_hub": False,
  "overwrite_output_dir": True,
  "output_dir": "/kaggle/working/mms_feruza_out",
  "report_to": ["tensorboard"],

  "dataset_name": "/kaggle/working/fz_ds",
  "audio_column_name": "audio",
  "text_column_name": "text",
  "train_split_name": "train",
  "eval_split_name": "train",
  "override_speaker_embeddings": False,

  "full_generation_sample_text": "Буғдой майдонида қўнғир занг касаллиги кўринди.",
  "max_duration_in_seconds": 12,
  "min_duration_in_seconds": 1.5,
  "max_tokens_length": 500,

  "model_name_or_path": "/kaggle/working/mms_uzb_disc",
  "preprocessing_num_workers": 4,

  "do_train": True,
  "num_train_epochs": 60,
  "gradient_accumulation_steps": 2,
  "gradient_checkpointing": False,
  "per_device_train_batch_size": 8,
  "learning_rate": 2e-5,
  "adam_beta1": 0.8, "adam_beta2": 0.99,
  "warmup_ratio": 0.01,
  "group_by_length": False,

  "do_eval": True, "eval_steps": 200,
  "per_device_eval_batch_size": 8, "max_eval_samples": 8,
  "do_step_schedule_per_epoch": True,

  "weight_disc": 3, "weight_fmaps": 1, "weight_gen": 1,
  "weight_kl": 1.5, "weight_duration": 1, "weight_mel": 35,
  "fp16": True, "seed": 456
}
open("/kaggle/working/finetune-hf-vits/config_uzb.json","w").write(json.dumps(cfg, indent=2))
print("wrote config_uzb.json")
""")

md("## 6. Fine-tune")
code("""
%cd /kaggle/working/finetune-hf-vits
import subprocess
r = subprocess.run(["accelerate","launch","run_vits_finetuning.py","./config_uzb.json"])
if r.returncode != 0: raise RuntimeError("VITS finetuning failed")
print("training done")
""")

md("## 7. Inference — agriculture sentences in the fine-tuned FeruzaSpeech voice")
code("""
import os, numpy as np, scipy.io.wavfile as wav, torch
from transformers import VitsModel, AutoTokenizer
M="/kaggle/working/mms_feruza_out"
model=VitsModel.from_pretrained(M); tok=AutoTokenizer.from_pretrained(M)
sr=model.config.sampling_rate
sents={
 "greet":"Салом! Помидор баргида сариқ доғлар пайдо бўлди. Нима қилишим керак?",
 "rust":"Буғдой майдонида қўнғир занг касаллиги кўринди.",
 "dose":"Фунгицидни йигирма беш фоиз концентрацияда, бир гектарга икки литр солинг.",
 "greet2":"Ассалому алайкум! Мен овозли ёрдамчисиман.",
}
os.makedirs("/kaggle/working/samples", exist_ok=True)
for k,t in sents.items():
    inp=tok(t, return_tensors="pt")
    with torch.no_grad(): y=model(**inp).waveform[0].cpu().numpy()
    y=(y/max(1e-9,np.abs(y).max())*0.95*32767).astype("int16")
    p=f"/kaggle/working/samples/ft_{k}.wav"; wav.write(p,sr,y); print("wrote",p,f"{len(y)/sr:.1f}s")
from IPython.display import Audio, display
for k in sents: display(Audio(f"/kaggle/working/samples/ft_{k}.wav"))
""")

nb={"cells":CELLS,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python"},"accelerator":"GPU"},"nbformat":4,"nbformat_minor":5}
out=os.path.join(os.path.dirname(__file__),"mms_uzbek_feruza_finetune.ipynb")
json.dump(nb,open(out,"w"),indent=1)
print("wrote",out,len(CELLS),"cells")
