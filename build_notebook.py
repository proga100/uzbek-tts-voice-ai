"""Generate feruzaspeech_mms_finetune.ipynb.

Edit the cell sources below, then run `python build_notebook.py` to regenerate the
notebook. Keeping the source here (plain Python strings) avoids hand-escaping JSON.
"""
import json
import os

CELLS = []


def md(text: str) -> None:
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text: str) -> None:
    CELLS.append({
        "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    })


md("""
# FeruzaSpeech → MMS-TTS Uzbek fine-tune (research proof)

Fine-tunes `facebook/mms-tts-uzb-script_latin` (VITS) on the clean single-speaker
**FeruzaSpeech** corpus to prove a custom Uzbek voice can be trained.

**Research / non-commercial only** — FeruzaSpeech is academic-license and MMS-TTS is
CC-BY-NC. The output cannot ship commercially; it only demonstrates the pipeline.

**Before running:** GPU on (Settings → Accelerator → GPU), accept the FeruzaSpeech terms
on HF, and add your HF token as a Kaggle secret named `HF_TOKEN`.
""")

md("## 1. Install dependencies + the fine-tuning recipe")

code("""
# VITS/MMS fine-tuning tooling (ylacombe/finetune-hf-vits) + audio deps.
!pip -q install "transformers>=4.41.0" "datasets>=2.19.0" "accelerate>=0.30.0" \\
    soundfile librosa scipy "huggingface_hub>=0.23.0"

import os
os.chdir("/kaggle/working")
if not os.path.exists("finetune-hf-vits"):
    !git clone -q https://github.com/ylacombe/finetune-hf-vits.git
os.chdir("/kaggle/working/finetune-hf-vits")

# Build the monotonic alignment CUDA/C extension the VITS training loss needs.
%cd monotonic_align
!mkdir -p monotonic_align && python setup.py build_ext --inplace -q
%cd /kaggle/working/finetune-hf-vits
print("setup done:", os.getcwd())
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
## 3. Load FeruzaSpeech and build a small training subset

We take a subset for a fast proof (raise `SUBSET_HOURS` later for quality). The dataset
must end up with an `audio` column (16 kHz) and a `text` column (Latin transcript).
""")

code("""
from datasets import load_dataset, Audio, DatasetDict

# FeruzaSpeech is gated; token auth (step 2) lets this through.
# ADJUST: split names / column names if the first run errors — print(raw) to inspect.
raw = load_dataset("k2speech/FeruzaSpeech", token=HF_TOKEN)
print(raw)

split = "train" if "train" in raw else list(raw.keys())[0]
ds = raw[split]

# ADJUST: pick the Latin-transcript column. FeruzaSpeech ships Latin + Cyrillic.
TEXT_COL = "text_latin" if "text_latin" in ds.column_names else (
    "latin" if "latin" in ds.column_names else "text")
AUDIO_COL = "audio"  # ADJUST if the audio column is named differently

# Keep a small, fast subset for the proof (~30 min of audio).
SUBSET = min(400, len(ds))
ds = ds.shuffle(seed=42).select(range(SUBSET))

# Normalise columns to: audio (16k) + text
ds = ds.cast_column(AUDIO_COL, Audio(sampling_rate=16000))
keep = {AUDIO_COL: "audio", TEXT_COL: "text"}
ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
ds = ds.rename_columns({k: v for k, v in keep.items() if k != v})

split_ds = ds.train_test_split(test_size=0.05, seed=42)
data = DatasetDict(train=split_ds["train"], eval=split_ds["test"])
data.save_to_disk("/kaggle/working/feruza_subset")
print(data)
print("example text:", data["train"][0]["text"][:120])
""")

md("""
## 4. Convert the base checkpoint (add the discriminator)

MMS-TTS checkpoints ship without the GAN discriminator needed for fine-tuning; the recipe's
converter downloads the base and adds it.
""")

code("""
# ADJUST: --language_code is the MMS code for Uzbek Latin. If this errors, check the
# repo README for the exact code (e.g. 'uzb-script_latin').
!python convert_original_discriminator_checkpoint.py \\
    --language_code uzb-script_latin \\
    --pytorch_dump_folder_path /kaggle/working/mms-uzb-base
print("base+discriminator ready")
""")

md("## 5. Write the training config")

code("""
import json

config = {
    "project_name": "mms-uzb-feruza",
    "push_to_hub": False,
    "overwrite_output_dir": True,
    "output_dir": "/kaggle/working/mms-uzb-feruza",

    "model_name_or_path": "/kaggle/working/mms-uzb-base",

    "dataset_name": "/kaggle/working/feruza_subset",
    "audio_column_name": "audio",
    "text_column_name": "text",
    "train_split_name": "train",
    "eval_split_name": "eval",

    "max_duration_in_seconds": 20.0,
    "min_duration_in_seconds": 1.0,

    "do_train": True,
    "do_eval": True,
    "num_train_epochs": 20,                 # proof; raise for quality
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "learning_rate": 2e-4,
    "warmup_ratio": 0.1,
    "fp16": True,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "save_total_limit": 1,
    "logging_steps": 10,

    "weight_disc": 3.0,
    "weight_fmaps": 1.0,
    "weight_gen": 1.0,
    "weight_kl": 1.5,
    "weight_mel": 35.0,
    "weight_duration": 1.0,

    "do_step_schedule_per_epoch": True,
    "seed": 42,
}
with open("/kaggle/working/finetune_config.json", "w") as f:
    json.dump(config, f, indent=2)
print("wrote finetune_config.json")
""")

md("## 6. Train")

code("""
# Single GPU is fine for the proof. ~20-40 min depending on subset size.
!accelerate launch run_vits_finetuning.py /kaggle/working/finetune_config.json
""")

md("## 7. Synthesize an Uzbek sample in the fine-tuned voice")

code("""
import torch, scipy.io.wavfile
from transformers import VitsModel, AutoTokenizer
from IPython.display import Audio as PlayAudio

MODEL_DIR = "/kaggle/working/mms-uzb-feruza"
model = VitsModel.from_pretrained(MODEL_DIR)
tok = AutoTokenizer.from_pretrained(MODEL_DIR)

text = "Salom! Men ovozli yordamchisiman. Sizga qanday yordam bera olaman?"
inputs = tok(text, return_tensors="pt")
with torch.no_grad():
    wav = model(**inputs).waveform[0].cpu().numpy()

sr = model.config.sampling_rate
scipy.io.wavfile.write("/kaggle/working/sample.wav", sr, wav)
print("wrote /kaggle/working/sample.wav @", sr, "Hz")
PlayAudio(wav, rate=sr)
""")

md("""
## Next steps

- This proves training works. Raise `SUBSET` (step 3) and `num_train_epochs` (step 5) for
  better quality.
- **Do not ship this voice commercially** — research/non-commercial license. For production:
  record your own narrator + fine-tune **Piper (MIT)** the same way, then serve behind the
  app's TTS provider interface.
""")

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = os.path.join(os.path.dirname(__file__), "feruzaspeech_mms_finetune.ipynb")
with open(out, "w") as f:
    json.dump(nb, f, indent=1)
print("wrote", out, "with", len(CELLS), "cells")
