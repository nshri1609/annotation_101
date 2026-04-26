import sys
import logging
logging.basicConfig(level=logging.DEBUG)

import torchaudio
if not hasattr(torchaudio, 'AudioMetaData'):
    torchaudio.AudioMetaData = type('AudioMetaData', (), {})
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ["soundfile", "sox"]

from videoannotator.registry import get_pipeline_loader

loader = get_pipeline_loader()
pipelines = loader.load_all_pipelines()
print("Loaded classes:", len(pipelines.keys()))
