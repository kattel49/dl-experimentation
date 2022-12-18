import whisper
import sys
from audio_recorder import run_threads

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch

print(torch.cuda.is_available())

filename = "output.wav"

run_threads()

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
image.save("trial.png")

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