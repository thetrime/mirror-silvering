from gpt_sovits.api import GPTSoVITS

tts = GPTSoVITS()
wav = tts.generate(#ref_text="The thing is, I dont really need an alarm. My internal alarm is really really good. Like, if I know I have to wake up at eight",
                   target_text="No body of water is safe without a lifeguard; that is true. That includes bathtubs.",
                   ref_audio_path="ref.wav",
                   output_path="sovits.wav")