import whisper
from audio_recorder import run_threads

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch

filename = "output.wav"
# capture audio
run_threads(filename)

# load and process the audio
model = whisper.load_model('medium').to('cuda')
model.eval()

audio = whisper.load_audio(filename)
audio = whisper.pad_or_trim(audio)
mel = whisper.log_mel_spectrogram(audio).to('cuda')

options = whisper.DecodingOptions(task='translate')

result_ = whisper.decode(model, mel, options)

print(result_.text, end="\n\n")

with open("transcribe.txt", "w+") as f:
    f.writelines([result_.text])

try:
    model_id = "stabilityai/stable-diffusion-2-1-base"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16)
    pipe = pipe.to('cuda')
    pipe.enable_attention_slicing()
    image = pipe(result_.text).images[0]
    image.save("trial.png")
except torch.cuda.OutOfMemoryError:
    print("Error: Cuda out of Memory")
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