import os
import threading
import pyaudio
import wave
import time

frames = []

sample_format = pyaudio.paInt32
fs = 44100
channels = 2
CHUNK=1024

kill_threads = False

def record_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=sample_format,
        channels=channels,
        rate=fs,
        frames_per_buffer=CHUNK,
        input=True)
    while True:
        if kill_threads:
            return
        data = stream.read(CHUNK)
        frames.append(data)



def save_audio(filename="output.wav"):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b"".join(frames))
    wf.close

def timer_thread():
    x = 0
    while True:
        if kill_threads:
            print(f"{x} seconds of audio recorded")
            return
        time.sleep(0.5)
        x += 0.5

class RecordingThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    
    def run(self):
        record_audio()

class TimingThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        timer_thread()


def run_threads(filename=None):
    #recording_thread = threading.Thread(target=record_audio)
    #timing_thread = threading.Thread(target=timer_thread)
    recording_thread = RecordingThread()
    timing_thread = TimingThread()
    recording_thread.start()
    timing_thread.start()

    user_input = input("Press any key to stop recording audio\n")
    global kill_threads

    if user_input:
        print(f"Hi this is user input: {user_input}")
        kill_threads = True
    
    recording_thread.join()
    timing_thread.join()

    if filename is not None:
        save_audio(filename=filename)
    else:
        save_audio()


if __name__ == "__main__":
    run_threads()