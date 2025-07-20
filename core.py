import soundfile as sf
import numpy as np
import webrtcvad
import atexit
import whisper  # or use openai.Audio.transcribe for remote
import sys
import resampy
import re
from llama_cpp import Llama
import asyncio
from huggingface_hub import hf_hub_download, snapshot_download

FRAME_DURATION = 30  # in ms
VAD_MODE = 2  # 0–3, 3 = aggressive
SILENCE_MS = 1000

SYSTEM_PROMPT = (
    "You are a friendly and curious assistant who always responds in a witty, British tone."
)


model_path = hf_hub_download(repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF", filename="mistral-7b-instruct-v0.2.Q5_K_S.gguf")

llm = Llama(
    model_path=model_path,
    n_ctx=32768,
    n_threads=8,
    n_gpu_layers=35
)

def read_audio(filename):
    audio, sample_rate = sf.read(filename)
    if len(audio.shape) > 1:  # Stereo -> mono
        audio = audio.mean(axis=1)
    return audio, sample_rate

def pcm_encode(audio, sample_rate):
    """Convert float32 numpy array to PCM16 bytes"""
    audio_int16 = np.int16(audio * 32767)
    return audio_int16.tobytes()

def split_audio_with_vad(audio, sample_rate):
    target_rate = 16000
    if sample_rate != target_rate:
        audio = resampy.resample(audio, sample_rate, target_rate)
        sample_rate = target_rate

    vad = webrtcvad.Vad(VAD_MODE)
    frame_len = int(target_rate * FRAME_DURATION / 1000)  # samples
    frame_bytes = frame_len * 2  # int16 = 2 bytes

    audio_pcm = pcm_encode(audio, target_rate)
    frames = [audio_pcm[i:i+frame_bytes]
              for i in range(0, len(audio_pcm), frame_bytes)]
    chunks, buffer, silence = [], [], 0
    threshold = int(SILENCE_MS / FRAME_DURATION)  # e.g. 300ms silence

    for frame in frames:
        if len(frame) != frame_bytes:
            break  # drop incomplete frame
        is_speech = vad.is_speech(frame, target_rate)
        buffer.append(frame)
        silence = silence+1 if not is_speech else 0
        if silence >= threshold:
            chunks.append(b''.join(buffer[:-silence]))
            buffer = buffer[-silence:]
            silence = 0

    if buffer:
        chunks.append(b''.join(buffer))

    return chunks

def pcm_to_float32(pcm_bytes):
    audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    return audio_np / 32768.0

def transcribe_chunks(chunks, sample_rate, whisper_model=None):
    all_text = []
    for i, chunk in enumerate(chunks):
        audio_np = pcm_to_float32(chunk)
        # Whisper expects audio as float32 np.array
        result = whisper_model.transcribe(audio_np, language='en', fp16=False, task='transcribe')
        print(f"[chunk {i}] {result['text']}")
        all_text.append(result['text'].strip())
    return ' '.join(all_text)

async def ask_llm_streaming(prompt: str):
    buffer = ""
    pattern = re.compile(r'([.,;!?。！？：\n])')  # punctuation that implies a boundary


    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]

    stream = llm.create_chat_completion(messages=messages, stream=True)

    for chunk in stream:
        delta = chunk["choices"][0]["delta"]
        token = delta.get("content", "")
        if not token:
            continue

        buffer += token

        # Check for natural break
        if pattern.search(token) or len(buffer) > 100:  # fail-safe: send if buffer too long
            yield buffer.strip()
            buffer = ""

    if buffer.strip():
        yield buffer.strip()


def ask_llm(prompt: str) -> str:

    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a witty British assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        max_tokens=512,
    )
    return response['choices'][0]['message']['content'].strip()

async def ask_llm_lsd_streaming(prompt: str):
    buffer = ""
    pattern = re.compile(r'([.,;!?。！？：\n])')  # punctuation that implies a boundary
    full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {prompt}\nAssistant:"
    stream = llm(full_prompt, temperature=0.7, max_tokens=512, stop=["User:", "System:", "\n\n"], stream=True)

    for chunk in stream:
        print(chunk)
        token = chunk["choices"][0]["text"]
        if not token:
            continue

        buffer += token

        # Check for natural break
        if pattern.search(token) or len(buffer) > 100:  # fail-safe: send if buffer too long
            yield buffer.strip()
            buffer = ""

    if buffer.strip():
        yield buffer.strip()


def main(filename):
    print(f"Reading audio: {filename}")
    audio, sample_rate = read_audio(filename)

    print("Splitting on pauses...")
    chunks = split_audio_with_vad(audio, sample_rate)
    print(f"Got {len(chunks)} chunks")

    print("Transcribing with Whisper...")
    model = whisper.load_model("base")  # or "tiny", "small", "medium", etc.
    text = transcribe_chunks(chunks, sample_rate, model)

    print("Final Transcription:")
    print(text)

    reply = ask_llm(text)
    print("LLM:", reply)


@atexit.register
def free_model():
    llm.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py your_audio.wav")
    else:
        main(sys.argv[1])
