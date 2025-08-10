# About GPT-SoVITS
GPT-SoVITS is not simple to package as a dependency (at least, not yet). I've included it as a git submodule so we can continue to track the upstream progress.

In gpt_sovits there's a kind of wrapper that I use to abstract it, but the other thing you need is to install all the GPT-SoVITS dependencies as well. That is, to install the deps for this project, you must
```
pip install -r requirements.txt
pip install -r extern/GPT-SoVITS/requirements.txt
```
These require a few tools to be pre-installed: ffmpeg, cmake, unzip

You need a reference audio file in ref.wav

## Next steps
   * Stream the audio
   * Gather the MFCC and predict the viseme for each chunk
   * Apply the viseme at the UI and play the audio
   * Preprocess the reference audio and allow using more than one
   * Experiment with different SoVITS models?