import speech_recognition as sr
import soundfile as sf
import sounddevice as sd
import numpy as np
import os
import threading
from deepmultilingualpunctuation import PunctuationModel

model = PunctuationModel()


def record_audio(samplerate=16000):
    # Initialize an empty list to hold the recording frames
    recording_frames = []
    # Flag to control the recording status
    is_recording = threading.Event()
    is_recording.set()

    def callback(indata, frames, time, status):
        # This callback function will be called by sounddevice for each audio block
        if is_recording.is_set():
            recording_frames.append(indata.copy())
        else:
            raise sd.CallbackStop

    # Start recording in a non-blocking manner
    with sd.InputStream(samplerate=samplerate, channels=1, dtype='float32', callback=callback):
        input("Press Enter to stop recording...")
        # When Enter is pressed, clear the recording flag to stop the callback
        is_recording.clear()

    # Concatenate all recorded frames
    recording = np.concatenate(recording_frames, axis=0)
    return recording


def play_audio(audio, samplerate=16000):
    sd.play(audio, samplerate)
    sd.wait()  # Wait for playback to finish


# Record audio
audio_clip = record_audio()

# Play back the recorded audio
# print("Playing back the recorded audio...")
# play_audio(audio_clip)


wav_file = 'recorded_audio.wav'
sf.write(wav_file, audio_clip, 16000)

recognizer = sr.Recognizer()

# Replace 'your_audio_file.wav' with the path to your audio file
audio_file_path = 'recorded_audio.wav'

# Use the audio file as the audio source
with sr.AudioFile(audio_file_path) as source:
    print("Reading audio file...")
    # Adjust the recognizer sensitivity to ambient noise
    recognizer.adjust_for_ambient_noise(source)
    # Read the audio file
    audio_data = recognizer.record(source)
    print("Recognizing...")
    try:
        # Recognize the speech in the audio file
        text = recognizer.recognize_google(audio_data)
        print(f"Audio file said: {text}")
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand the audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

# Grammer correction
result = model.restore_punctuation(text)
result = result.replace(',', '').replace('.', '').replace(":", '')
print(result)
