import asyncio
import websockets
import numpy as np

from whisper import load_model
from core import split_audio_with_vad, transcribe_chunks, ask_chat_streaming, ask_llm_streaming

chatMode = True
SAMPLE_RATE = 16000
CHUNK_DURATION_SEC = 5  # how much audio to collect before processing
BYTES_PER_SAMPLE = 2  # 16-bit PCM

whisper = load_model("base")

buffer = bytearray()


async def handle_client(websocket: any) -> None:
    print("Client connected")
    global buffer

    try:
        async for message in websocket:
            buffer.extend(message)

            # If we've buffered enough audio, process it
            if len(buffer) >= CHUNK_DURATION_SEC * SAMPLE_RATE * BYTES_PER_SAMPLE:
                print("Processing chunk...")

                # Convert raw bytes to numpy int16 array
                audio = np.frombuffer(buffer[:CHUNK_DURATION_SEC * SAMPLE_RATE * BYTES_PER_SAMPLE], dtype=np.int16).astype(np.float32) / 32768.0
                buffer = buffer[CHUNK_DURATION_SEC * SAMPLE_RATE * BYTES_PER_SAMPLE:]  # remove processed

                chunks = split_audio_with_vad(audio, SAMPLE_RATE)
                print(f"Processing {len(chunks)} chunks...")
                text = transcribe_chunks(chunks, SAMPLE_RATE, whisper)
                if text.strip():
                    if chatMode:
                        async for chunk in ask_chat_streaming(text):
                            await websocket.send(chunk)
                        await websocket.send("\n")
                    else:
                        async for chunk in ask_llm_streaming(text):
                            await websocket.send(chunk)
                        await websocket.send("\n")

                else:
                    print(f"No chunks received")

    except websockets.ConnectionClosed:
        print("Client disconnected")


async def main():
    async with websockets.serve(handle_client, "localhost", 8765):
        print("Server running on ws://localhost:8765")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
