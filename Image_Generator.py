import torch
from diffusers import DiffusionPipeline
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# Load the diffusion pipeline and move it to GPU (if available)
pipe = DiffusionPipeline.from_pretrained("FLUX.1-dev").to("cuda")

# Load the Ghibli-specific LoRA weights
pipe.load_lora_weights("flux-chatgpt-ghibli-lora")

# Load and preprocess the input image
input_image_path = "path/to/your/image.jpg"  # Replace with your image path
input_image = Image.open(input_image_path).convert("RGB")
input_image = input_image.resize((512, 512))  # Resize to model-compatible size

# Convert image to tensor and normalize to [-1, 1]
transform = transforms.ToTensor()
input_tensor = transform(input_image).unsqueeze(0).to("cuda")  # Add batch dimension
input_tensor = input_tensor * 2 - 1  # Normalize from [0, 1] to [-1, 1]

# Encode the input image into latent space using the VAE
vae = pipe.vae
with torch.no_grad():
    latents = vae.encode(input_tensor).latent_dist.sample()

# Define diffusion parameters
num_inference_steps = 50  # Number of denoising steps
strength = 0.8  # Controls how much the image is transformed (0.0 = no change, 1.0 = full transformation)
scheduler = pipe.scheduler
scheduler.set_timesteps(num_inference_steps)
timesteps = scheduler.timesteps

# Calculate the starting timestep based on strength
t_start = max(num_inference_steps - int(num_inference_steps * strength), 0)
starting_timestep = timesteps[t_start]

# Add noise to the latents to start the diffusion process
noise = torch.randn_like(latents)
noisy_latents = scheduler.add_noise(latents, noise, starting_timestep)

# Prepare the text prompt and embeddings for guidance
prompt = "a scene in Ghibli studio art style"  # Prompt to reinforce Ghibli style
tokenizer = pipe.tokenizer
text_inputs = tokenizer(
    prompt,
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt"
).to("cuda")
text_encoder = pipe.text_encoder
with torch.no_grad():
    text_embeddings = text_encoder(text_inputs.input_ids)[0]

# Generate unconditional embeddings for classifier-free guidance
uncond_input = tokenizer(
    [""],
    padding="max_length",
    max_length=tokenizer.model_max_length,
    return_tensors="pt"
).to("cuda")
with torch.no_grad():
    uncond_embeddings = text_encoder(uncond_input.input_ids)[0]

# Combine embeddings for classifier-free guidance
do_classifier_free_guidance = True
guidance_scale = 7.5  # Strength of prompt guidance
if do_classifier_free_guidance:
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

# Run the denoising loop
unet = pipe.unet
for t in timesteps[t_start:]:
    # Prepare the latent input for the UNet
    latent_model_input = torch.cat([noisy_latents] * 2) if do_classifier_free_guidance else noisy_latents
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

    # Apply classifier-free guidance
    if do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # Step to the next latent state
    noisy_latents = scheduler.step(noise_pred, t, noisy_latents).prev_sample

# Decode the final latents back to an image
with torch.no_grad():
    output_image = vae.decode(noisy_latents).sample

# Post-process the output image
output_image = (output_image / 2 + 0.5).clamp(0, 1)  # Denormalize to [0, 1]
output_image = output_image.cpu().permute(0, 2, 3, 1).numpy()[0]  # Reshape to HWC
output_image = (output_image * 255).astype(np.uint8)  # Convert to uint8

# Save or display the result
Image.fromarray(output_image).save("output_ghibli_style.png")