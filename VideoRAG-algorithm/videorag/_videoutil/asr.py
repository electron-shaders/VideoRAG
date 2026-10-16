import os
import logging

import av
import numpy as np
from tqdm import tqdm


def decode_segment_audio(audio_file, audio_format, sampling_rate=16000):
    """Decode our generated audio using its known container, not probing.

    Some valid MP3 segments are misidentified as VVC video by PyAV. Passing
    decoded mono samples to Whisper avoids its automatic format detection.
    """
    resampler = av.AudioResampler(format="s16", layout="mono", rate=sampling_rate)
    samples = []
    with av.open(audio_file, mode="r", format=audio_format) as container:
        if not container.streams.audio:
            raise ValueError(f"No audio stream in {audio_file}")
        for frame in container.decode(audio=0):
            frame.pts = None
            samples.extend(resampled.to_ndarray().reshape(-1) for resampled in resampler.resample(frame))
        samples.extend(frame.to_ndarray().reshape(-1) for frame in resampler.resample(None))
    if not samples:
        raise ValueError(f"No audio samples decoded from {audio_file}")
    return np.concatenate(samples).astype(np.float32) / 32768.0


def speech_to_text(video_name, working_dir, segment_index2name, audio_output_format):
    from faster_whisper import WhisperModel

    model = WhisperModel("Systran/faster-distil-whisper-large-v3")
    model.logger.setLevel(logging.WARNING)
    
    cache_path = os.path.join(working_dir, '_cache', video_name)
    
    transcripts = {}
    for index in tqdm(segment_index2name, desc=f"Speech Recognition {video_name}"):
        segment_name = segment_index2name[index]
        audio_file = os.path.join(cache_path, f"{segment_name}.{audio_output_format}")

        # if the audio file does not exist, skip it
        if not os.path.exists(audio_file):
            transcripts[index] = ""
            continue
        
        audio = decode_segment_audio(audio_file, audio_output_format)
        segments, info = model.transcribe(audio)
        result = ""
        for segment in segments:
            result += "[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text)
        transcripts[index] = result
    
    return transcripts
