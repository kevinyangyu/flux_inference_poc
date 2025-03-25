
## 
While trying to generate images using black-forest-labs/FLUX.1-dev I realized the model (fp16) was too large to be loaded into a single gpu. You could improve the speed of image generation if you handle text encoding first and either split the transformer model across multiple gpus or load multiple instances of the transformer. 

You can apply adapters to both the transformers as well as the text encoders

The inference pipeline breaks down the generation process into several stages:
1. Text Embedding Generation
2. Transformer Processing
3. VAE Decoding and Image Generation


## Usage

```bash
python run_inference_pipeline.py
```

## Arguments

- `--model_path`: Path to the Flux model directory (required)
- `--adapter_path`: Path to the adapter weights directory (optional)
- `--prompt_directory`: Directory containing prompt files (required)
- `--adapter_weight_name`: Name of the adapter weight file (default: 'photo.safetensors')
- `--gpu_memory`: GPU memory limit per device (default: '24GB')
- `--height`: Output image height (default: 768)
- `--width`: Output image width (default: 768)
- `--output_dir`: Output directory for generated images (default: 'test_batch')
- `--num_inference_steps`: Number of inference steps (default: 55)
- `--guidance_scale`: Guidance scale for inference (default: 3.0)
- `--vae_scale_factor`: VAE scaling factor


### 1. Text encoders
- Loads text encoders and tokenizers
- Processes input prompts
- Cleans up memory after generation

### 2. Transformer Processing
- Loads the Flux transformer model across devices
- Applies LoRA adapters to the transformer model
- Processes the embeddings to generate latents

### 3. VAE Decoding
- Decodes latents into images using the VAE
- Performs post-processing
- Saves the final generated images

## Input Format

The pipeline accepts prompt files in JSON format with either:
- Both "clip_l" and "T5" fields for dual-encoder setups or a single prompt as "description"
