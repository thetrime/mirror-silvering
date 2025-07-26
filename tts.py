import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="mps")

text = "No body of water is safe without a lifeguard, that is true. That includes bathtubs"
wav = model.generate(text, audio_prompt_path="ref.wav")
ta.save("test-2.wav", wav, model.sr)