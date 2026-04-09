import torch
from diffusers import DiffusionPipeline
from diffusers.utils import load_image

# Use torch_dtype instead of dtype
pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B", 
    torch_dtype=torch.bfloat16, 
    device_map="cuda"
)

pipe.enable_model_cpu_offload()

prompt = "Replace the face of the person with a different realistic human face, keeping the same pose, lighting, camera angle, hairstyle, and background. Maintain natural skin texture and consistent shadows. The new face should blend seamlessly with the body and environment, photorealistic, high detail, no artifacts."
input_image = load_image("./local-image.png")

image = pipe(image=input_image, prompt=prompt).images[0]

image.save("output.png")