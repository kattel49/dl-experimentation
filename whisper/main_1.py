import whisper
import pyaudio
import wave
import sys


from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch

print(torch.cuda.is_available())

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
i = 0
while True:
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

    model = whisper.load_model('base').to('cuda')

    result = model.transcribe(filename)
    print(result['text'], result['language'])


    audio = whisper.load_audio(filename)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to('cuda')

    options = whisper.DecodingOptions(task='translate')

    result_ = whisper.decode(model, mel, options)
    print(result_.text)


    model_id = "stabilityai/stable-diffusion-2-1-base"

    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    image = pipe(result_.text).images[0]

    image.save(f"trial_{i}.png")
    i+=1