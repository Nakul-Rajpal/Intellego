import speech_recognition as sr
import soundfile as sf
import sounddevice as sd
import numpy as np
import os
import threading
from happytransformer import HappyTextToText, TTSettings
import time

stop_recording = False
verbose = False
wait_time = 5


def check_similarity(spoken, correct):
    correct_lower = correct.lower()
    correct_lower = correct_lower.replace(',', '').replace('.', '').replace(":", '')
    return {
        "spoken": spoken,
        "correct": correct,
        "is_correct": correct_lower == spoken.lower()
    }


def record_audio(samplerate=16000):
    # Initialize an empty list to hold the recording frames
    recording_frames = []
    # Flag to control the recording status
    print("Loading recorder...")
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
        time.sleep(wait_time)
        is_recording.clear()

    # Concatenate all recorded frames
    recording = np.concatenate(recording_frames, axis=0)
    return recording


def play_audio(audio, samplerate=16000):
    sd.play(audio, samplerate)
    sd.wait()  # Wait for playback to finish


if __name__ == '__main__':
    print("Starting Intellego")
    # Record audio

    print("Initializing HappyTextToText")
    model_dir = "model"
    if os.path.exists(model_dir) and os.path.isdir(model_dir):
        happy_tt = HappyTextToText(model_type="T5", model_name="model/")
    else:
        os.mkdir(model_dir)
        happy_tt = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
        happy_tt.save("model/")
    print("Done initializing HappyTextToText")

    stop_recording = False
    while not stop_recording:
        audio_clip = record_audio()

        # Play back the recorded audio
        # print("Playing back the recorded audio...")
        # play_audio(audio_clip)

        audio_file_path = "recorded_audio.wav"
        sf.write(audio_file_path, audio_clip, 16000)

        recognizer = sr.Recognizer()
        # Replace 'your_audio_file.wav' with the path to your audio file
        # Use the audio file as the audio source
        with sr.AudioFile(audio_file_path) as source:
            if verbose: print("Reading audio file...")
            # Adjust the recognizer sensitivity to ambient noise
            recognizer.adjust_for_ambient_noise(source)
            # Read the audio file
            audio_data = recognizer.record(source)
            if verbose: print("Recognizing...")
            try:
                # Recognize the speech in the audio file
                text = recognizer.recognize_google(audio_data)
                if verbose: print(f"Audio file said: {text}")
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand the audio")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")

        args = TTSettings(num_beams=5, min_length=1)

        # Add the prefix "grammar: " before each input
        result = happy_tt.generate_text("grammar: " + text, args=args)
        if text == "stop" or text == "stop recording":
            print("User stopped recording")
            break
        else:
            similarity = check_similarity(text, result.text)

            if similarity["is_correct"]:
                print("{} ✅".format(similarity["correct"]))
            else:
                print("{} ❌ {} ✅".format(similarity["spoken"], similarity["correct"]))
