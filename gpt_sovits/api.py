import sys
from pathlib import Path
import os
import nltk
from core import whisper
import torch

nltk.download('averaged_perceptron_tagger_eng')

# Add GPT-SoVITS to the module path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "extern" / "GPT-SoVITS"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "extern" / "GPT-SoVITS" / "GPT_SoVITS"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "extern" / "GPT-SoVITS" / "GPT_SoVITS" / "eres2net"))

from gpt_sovits.models import init_model_env, init_checkpoint

# Versions that work: v1, v2, v4
# Versions that don't: v2Pro, v2ProPlus, v3 (all missing auxiliary stuff that uses hard-coded paths)
version = "v2Pro"
sovits = {"v1": (None, "s2G488k.pth"),
          "v2": ("gsv-v2final-pretrained", "s2G2333k.pth"),
          "v3": (None, "s2Gv3.pth",),
          "v4": ("gsv-v4-pretrained", "s2Gv4.pth"),
          "v2Pro": ("v2Pro", "s2Gv2Pro.pth"),
          "v2ProPlus": ("v2Pro", "s2Gv2ProPlus.pth")}

gpt = {"v1": (None, "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"),
       "v2": ("gsv-v2final-pretrained", "s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"),
       "v3": (None, "s1v3.ckpt"),
       "v4": (None, "s1v3.ckpt"),
       "v2Pro": (None, "s1v3.ckpt"),
       "V2ProPlus": (None, "s1v3.ckpt")}


# Must run before importing GPT-SoVITS internals
init_model_env({
    "cnhubert_base_path": (
        "chinese-hubert-base",
        ("config.json", "preprocessor_config.json", "pytorch_model.bin")
    ),
    "bert_path": (
        "chinese-roberta-wwm-ext-large",
        ("config.json", "tokenizer.json", "pytorch_model.bin")
    )})

init_checkpoint({
    "gpt_path": gpt[version],
    "sovits_path": sovits[version]
})


# This all hacks together the v4 vocoder. It's _not_ pretty
if version == "v4":
    init_checkpoint({
        "vocoder_path": ("gsv-v4-pretrained", "vocoder.pth")
    })
if version == "v3":
    init_model_env({
        "vocoder_path": ("models--nvidia--bigvgan_v2_24khz_100band_256x", ("config.json", "bigvgan_generator.pt"))
    })

_orig_torch_load = torch.load
def patched_torch_load(path, *args, **kwargs):
    if isinstance(path, str) and path.endswith("vocoder.pth"):
        return _orig_torch_load(os.environ["vocoder_path"], *args, **kwargs)
    else:
        return _orig_torch_load(path, *args, **kwargs)
torch.load = patched_torch_load
# End upgly hack

from GPT_SoVITS.inference_webui import (
    change_gpt_weights,
    change_sovits_weights,
    get_tts_wav,
)
from tools.i18n.i18n import I18nAuto

import librosa
import soundfile as sf
import numpy as np
import tempfile

def prepare_audio(path, max_seconds=10.0, sr_target=None, top_db=30) -> str:
    wav, sr = librosa.load(path, sr=sr_target)
    max_samples = int(max_seconds * sr)

    if len(wav) <= max_samples:
        trimmed = wav
    else:
        # Find non-silent intervals
        nonsilent_intervals = librosa.effects.split(wav, top_db=top_db)

        center_sample = len(wav) // 2

        # Find silent boundaries (edges of non-silent intervals)
        silent_boundaries = []

        # Add start and end edges as silence too
        prev_end = 0
        for start, end in nonsilent_intervals:
            # silence before this nonsilent segment
            if start > prev_end:
                silent_boundaries.append( (prev_end, start) )
            prev_end = end
        # silence after last nonsilent segment
        if prev_end < len(wav):
            silent_boundaries.append( (prev_end, len(wav)) )

        # Find silent boundary closest to center sample with enough length for trimming
        suitable_silence = None
        for start, end in silent_boundaries:
            length = end - start
            if length >= max_samples:
                # Check if center_sample falls within this silence, or nearby
                if start <= center_sample <= end:
                    suitable_silence = (start, end)
                    break
        # fallback: pick the silence with max length closest to center
        if not suitable_silence and silent_boundaries:
            silent_boundaries.sort(key=lambda x: (abs((x[0]+x[1])//2 - center_sample), -(x[1]-x[0])))
            suitable_silence = silent_boundaries[0]

        if suitable_silence:
            # Place the max_seconds window within this silent region, centered near center_sample
            silence_start, silence_end = suitable_silence
            silence_center = (silence_start + silence_end) // 2
            start = silence_center - max_samples // 2
            start = max(silence_start, start)
            if start + max_samples > silence_end:
                start = silence_end - max_samples
        else:
            # fallback to simple center trim
            start = center_sample - max_samples // 2

        start = max(0, start)
        end = start + max_samples
        if end > len(wav):
            end = len(wav)
            start = end - max_samples

        trimmed = wav[start:end]

    trimmed = trimmed.astype(np.float32)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, trimmed, sr, format="WAV")
    tmp.close()
    return tmp.name

class GPTSoVITS:
    def __init__(self):
        self.i18n = I18nAuto()

        # Load the models
        change_gpt_weights(gpt_path=os.environ["gpt_path"])
        change_sovits_weights(sovits_path=os.environ["sovits_path"])

    def generate(
            self,
            ref_audio_path: str,
            target_text: str,
            target_language: str = "English",
            ref_language: str = "English",
            output_path: str = "output.wav",
            top_p=1,
            temperature=1,
    ):
        # Translate languages
        ref_lang = self.i18n(ref_language)
        tgt_lang = self.i18n(target_language)

        prepared_audio = prepare_audio(ref_audio_path)
        ref_text = whisper.transcribe(prepared_audio)["text"].strip()
        print(f"Calculated {ref_text}")
        print(f"Prepared audio: {prepared_audio}")
        try:
            # Run synthesis
            result = list(
                get_tts_wav(
                    ref_wav_path=prepared_audio,
                    prompt_text=ref_text,
                    prompt_language=ref_lang,
                    text=target_text,
                    text_language=tgt_lang,
                    top_p=top_p,
                    temperature=temperature,
                )
            )
        finally:
            os.remove(prepared_audio)

        if not result:
            raise RuntimeError("Synthesis returned no results")

        sampling_rate, audio = result[-1]
        sf.write(output_path, audio, sampling_rate)
        return output_path
