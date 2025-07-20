import asyncio
import websockets
import sounddevice as sd
import numpy as np

SAMPLE_RATE = 16000
CHUNK_SIZE = 1024  # samples per frame
SERVER_URI = "ws://localhost:8765"

async def stream_audio():
    async with websockets.connect(SERVER_URI) as websocket:
        print("Connected to server")

        def callback(indata, frames, time, status):
            if status:
                print("Mic status:", status)
            loop.call_soon_threadsafe(asyncio.create_task, websocket.send(indata.tobytes()))

        with sd.InputStream(
                channels=1,
                samplerate=SAMPLE_RATE,
                dtype='int16',
                blocksize=CHUNK_SIZE,
                callback=callback
        ):
            print("Streaming audio...")
            while True:
                try:
                    reply = await websocket.recv()
                    print("LLM:", reply)
                except websockets.ConnectionClosed:
                    print("Server disconnected")
                    break

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(stream_audio())
