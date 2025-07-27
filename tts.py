from gpt_sovits.api import GPTSoVITS

tts = GPTSoVITS()
wav = tts.generate(target_text="No body of water is safe without a lifeguard; that is true. That includes bathtubs.",
                   ref_audio_path="ref.wav",
                   output_path="sovits.wav")