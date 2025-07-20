import soundfile as sf
import numpy as np
import webrtcvad
import atexit
import resampy
import re
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from collections.abc import AsyncGenerator

FRAME_DURATION = 30  # in ms
VAD_MODE = 2  # 0–3, 3 = aggressive
SILENCE_MS = 1000

SYSTEM_PROMPT = (
    "You are a friendly and curious assistant who always responds in a witty, British tone. Your name is Nigel."
)

CHUNK_DELINEATION = re.compile(r'([.,;!?。！？：\n])')


model_path = hf_hub_download(repo_id="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
                             filename="mistral-7b-instruct-v0.2.Q5_K_S.gguf")

llm = Llama(
    model_path=model_path,
    n_ctx=32768,
    n_threads=8,
    n_gpu_layers=35,
    chat_format="llama-3"
)


def read_audio(filename):
    audio, sample_rate = sf.read(filename)
    if len(audio.shape) > 1:  # Stereo -> mono
        audio = audio.mean(axis=1)
    return audio, sample_rate


def pcm_encode(audio):
    """Convert float32 numpy array to PCM16 bytes"""
    audio_int16 = np.int16(audio * 32767)
    return audio_int16.tobytes()


def split_audio_with_vad(audio, sample_rate):
    target_rate = 16000
    if sample_rate != target_rate:
        audio = resampy.resample(audio, sample_rate, target_rate)
    vad = webrtcvad.Vad(VAD_MODE)
    frame_len = int(target_rate * FRAME_DURATION / 1000)  # samples
    frame_bytes = frame_len * 2  # int16 = 2 bytes

    audio_pcm = pcm_encode(audio)
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


async def chunked_stream(stream, get_token) -> AsyncGenerator[str, None]:
    buffer = ""
    async for chunk in stream:
        token = get_token(chunk)
        if not isinstance(token, str):
            continue

        buffer += token
        if CHUNK_DELINEATION.search(token) or len(buffer) > 100:
            yield buffer.strip()
            buffer = ""

    if buffer.strip():
        yield buffer.strip()


async def ask_chat_streaming(prompt: str):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    stream = llm.create_chat_completion(messages=messages, stream=True)
    return chunked_stream(stream, lambda c: c["choices"][0]["delta"].get("content"))


async def ask_llm_streaming(prompt: str):
    full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {prompt}\nAssistant:"
    stream = llm(full_prompt, temperature=0.7, max_tokens=512,
                 stop=["User:", "System:", "\n\n"], stream=True)
    return chunked_stream(stream, lambda c: c["choices"][0]["text"])


@atexit.register
def free_model():
    llm.close()
