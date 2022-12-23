from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch

with open('transcribe.txt', 'r') as f:
    result_ = f.readline()

model_id = "stabilityai/stable-diffusion-2"
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16)
pipe = pipe.to('cuda')
pipe.enable_attention_slicing()
image = pipe(result_).images[0]
image.save("trial.png")