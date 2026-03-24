import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image

# switch to "mps" for apple devices
pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", dtype=torch.bfloat16, device_map="cuda")

prompt = "Turn this cat into a dog"
idea_prompt = "a realistic human face, same pose and expression, anonymized identity, natural skin, no recognizable person"
input_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")

image = pipe(image=input_image, prompt=prompt).images[0]