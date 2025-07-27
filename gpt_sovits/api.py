import sys
from pathlib import Path
import os
import nltk
from io import BytesIO
import librosa
import soundfile as sf
import numpy as np


nltk.download('averaged_perceptron_tagger_eng')

# Add GPT-SoVITS to the module path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "extern" / "GPT-SoVITS"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "extern" / "GPT-SoVITS" / "GPT_SoVITS"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "extern" / "GPT-SoVITS" / "GPT_SoVITS" / "eres2net"))

from gpt_sovits.models import init_model_env, init_checkpoint

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
    "gpt_path": (
        None,
        "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
    ),
    "sovits_path": (
        None,
        "s2G488k.pth"
    ),
})

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

def prepare_audio(path, max_seconds=10.0) -> str:
    wav, sr = librosa.load(path, sr=None)
    max_samples = int(max_seconds * sr)
    if len(wav) > max_samples:
        start = (len(wav) - max_samples) // 2
        wav = wav[start:start + max_samples]

    wav = wav.astype(np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, wav, sr, format="WAV")
        return tmp.name  # Return the filename

class GPTSoVITS:
    def __init__(self):
        self.i18n = I18nAuto()

        # Load the models
        change_gpt_weights(gpt_path=os.environ["gpt_path"])
        change_sovits_weights(sovits_path=os.environ["sovits_path"])

    def generate(
            self,
            ref_audio_path: str,
            ref_text: str,
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
