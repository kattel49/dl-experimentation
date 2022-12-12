import whisper

import pyaudio
import wave
import sys

CHUNK = 1024
sample_format = pyaudio.paInt32
fs = 44100
seconds = 10
filename = "output.wav"
channels=2

p = pyaudio.PyAudio()

stream = p.open(format=sample_format,
            channels=channels,
            rate=fs,
            frames_per_buffer=CHUNK,
            input=True)

frames = []

record_audio=input("Do you want to record audio(y/n): ")

if record_audio == 'y':
    print("Started recording audio")
    for i in range(0, int(fs/CHUNK*seconds)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("Stopped recording audio")
    stream.stop_stream()
    stream.close()
    p.terminate()

    # save file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(sample_format))
    wf.setframerate(fs)
    wf.writeframes(b"".join(frames))
    wf.close()

model = whisper.load_model('medium').to('cuda')

result = model.transcribe(filename)
print(result['text'], result['language'])


audio = whisper.load_audio(filename)
audio = whisper.pad_or_trim(audio)
mel = whisper.log_mel_spectrogram(audio).to('cuda')

options = whisper.DecodingOptions(task='translate')

result_ = whisper.decode(model, mel, options)
print(result_.text)

# import whisper
# model = whisper.load_model('base').to('cuda')

# audio = whisper.load_audio('sound.flac')
# audio = whisper.pad_or_trim(audio)

# #make log-mel spectrogram
# mel = whisper.log_mel_spectrogram(audio).to('cuda')

# _, probs = model.detect_language(mel)
# print("probability of nepalese language: ", probs["ne"])
# # decode the audio
# options = whisper.DecodingOptions(task='transcribe')
# result = whisper.decode(model, mel, options)
# print(result.text)
# print(result.language)