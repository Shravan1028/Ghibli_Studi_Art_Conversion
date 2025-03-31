# Ghibli_Studi_Art_Conversion
This Code Converts Image into a Ghibli Studio art

# Ghibli Style Image Transformation with FLUX.1-dev and LoRA Weights

This project demonstrates how to transform pre-existing images into the iconic Ghibli Studio art style using the FLUX.1-dev diffusion model and Ghibli-specific LoRA weights. By leveraging the power of diffusion models and style-specific adaptations, you can apply the whimsical and artistic flair of Ghibli Studio to your own images.

---

## Model Downloads

To use this project, you need to download the following models from Hugging Face:

- **FLUX.1-dev Model**:  
  [Download here](https://huggingface.co/black-forest-labs/FLUX.1-dev/tree/main)  
  This is the base diffusion model used for image generation and transformation.

- **Ghibli LoRA Weights**:  
  [Download here](https://huggingface.co/openfree/flux-chatgpt-ghibli-lora/tree/main)  
  These LoRA (Low-Rank Adaptation) weights adapt the FLUX.1-dev model to generate images in the Ghibli Studio style.

**Note**: Ensure you download the necessary files from these repositories. For the FLUX.1-dev model, you typically need the model weights and configuration files. For the Ghibli LoRA weights, download the LoRA adapter files.

---

## Installation

Before running the code, install the required libraries using the following pip command:

```bash
pip install diffusers Pillow torchvision numpy
