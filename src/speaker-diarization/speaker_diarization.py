from dataclasses import dataclass

from manimlib import *

import numpy as np
from datasets import load_dataset
from pydub import AudioSegment
from scipy.io import wavfile



DATASET_NAME = "talkbank/callhome"
MAX_SECONDS = 10
COLORS = {
    0: GREY_B,
    1: BLUE,
    2: GREEN,
}
TEMP_FILE = "temp.wav"

@dataclass
class DiarizationSample:
    raw_array: np.ndarray
    timestamps: np.ndarray
    sample_rate: int
    speaker_ids: np.ndarray

def save_wav_file(file_name, sample_rate, audio_data):
    """
    Save a NumPy array as a .wav file.
    
    Parameters:
    file_name (str): Name of the output .wav file.
    sample_rate (int): The sample rate (samples per second).
    audio_data (numpy.ndarray): The audio data to be saved. Should be in the range -1.0 to 1.0 if float32.
    """
    # Ensure the data is in an appropriate format (e.g., int16 or float32)
    if audio_data.dtype != np.int16 and audio_data.dtype != np.float32:
        raise ValueError("Audio data should be of type int16 or float32, but is {}".format(audio_data.dtype))
    
    # Save the audio data as a .wav file
    wavfile.write(file_name, sample_rate, audio_data)

def load_sample(max_seconds: float = 30) -> DiarizationSample:
    ds = load_dataset(DATASET_NAME, "eng", split="data")
    audio = ds[0]["audio"]

    sample_rate = audio["sampling_rate"]
    waveform = audio["array"]
    timestamps_start = ds[0]["timestamps_start"]
    timestamps_end = ds[0]["timestamps_end"]
    speakers: str = ds[0]["speakers"]

    # Create a buffer for speaker ids -> strings
    speaker_ids = np.zeros_like(waveform, dtype=int)
    for start, end, speaker in zip(timestamps_start, timestamps_end, speakers):
        # Start and end are in seconds
        start = int(start * sample_rate)
        end = int(end * sample_rate)
        speaker_ids[start:end] = ord(speaker) - ord("A") + 1
    
    # Slice the waveform to match last end
    idx = int(sample_rate * max_seconds)
    waveform = waveform[:idx]
    speaker_ids = speaker_ids[:idx]
    timestamps = np.arange(len(waveform)) / sample_rate

    return DiarizationSample(
        raw_array=waveform.astype(np.float32),
        timestamps=timestamps,
        sample_rate=sample_rate,
        speaker_ids=speaker_ids,
    )


class SpeakerDiarization(Scene):
    def construct(self):
        # In this scene, we will:
        # - Load audio sample
        # - Create an axes or a box/rectangle where we'll plot the waveform
        # - Add a legend on top of the axes with the speaker ids (0 - No Voice, 1 - Speaker A, 2 - Speaker B)
        # - Animate the waveform, highlighting each speaker with a different color
        # - Add a sound to the scene
        # - Add a text to the scene with the title "Speaker Diarization"
        # - Move while shrinking the box-waveform to the left

        #################### START ####################

        # Load the sample
        waveform = load_sample(MAX_SECONDS)
        # Create a helper axes for plotting the waveform
        axes = Axes(
            x_range=(waveform.timestamps[0], waveform.timestamps[-1], 1),
            y_range=(waveform.raw_array.min(), waveform.raw_array.max(), 1),
            height=4,
            width=8,
        )
        axes.center()
        # Create a box to hold the waveform
        box = Rectangle(height=4, width=8.5)
        box.center()
        # Create a group to hold the box and curve
        plot = VGroup()
        plot.add(box)
        # Fade in the box
        self.play(FadeIn(box))
        # Create the waveform curve and set colors according to speaker ids
        waveform_curve = VMobject().set_points_as_corners(
            axes.c2p(*np.stack([waveform.timestamps, waveform.raw_array], axis=0))
        )
        colors = [COLORS[speaker_id] for speaker_id in waveform.speaker_ids]
        waveform_curve.set_color(colors)
        # Add the waveform curve to the plot
        plot.add(waveform_curve)
        # Animate the waveform curve
        self.play(
            ShowCreation(
                waveform_curve, 
                run_time=MAX_SECONDS,
                rate_func=linear,
            )
        )
        # Animate the plot shrinking and moving to the left
        self.play(plot.animate.scale(0.5))
        self.play(plot.animate.shift(4*LEFT))
