from gpt_sovits.api import GPTSoVITS
import soundfile as sf
import numpy as np
import time

tts = GPTSoVITS()

t0 = time.time()
chunks = []
for chunk in tts.generate(target_text="No body of water is safe without a lifeguard; that is true. That includes bathtubs.",
                          ref_audio_path="ref.wav"):
    chunks.append(chunk)
print(f"Total time: {time.time() - t0}")
sf.write("sovits-stream.wav", np.concatenate(chunks), samplerate=44100)
