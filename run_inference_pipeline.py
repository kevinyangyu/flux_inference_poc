import gc
import json
import torch
import argparse
from pathlib import Path
from datetime import datetime
from diffusers import FluxPipeline, FluxTransformer2DModel, AutoencoderKL
from diffusers.image_processor import VaeImageProcessor


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference pipeline with Flux model')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the Flux model directory')
    parser.add_argument('--adapter_path', type=str,
                      help='Path to the adapter weights directory')
    parser.add_argument('--prompt_directory', type=str, required=True,
                      help='Directory containing prompt files')
    parser.add_argument('--adapter_weight_name', type=str, default='photo.safetensors',
                      help='Name of the adapter weight file')
    parser.add_argument('--gpu_memory', type=str, default='24GB',
                      help='GPU memory limit per device')
    parser.add_argument('--height', type=int, default=768,
                      help='Output image height')
    parser.add_argument('--width', type=int, default=768,
                      help='Output image width')
    parser.add_argument('--output_dir', type=str, default='test_batch',
                      help='Output directory for generated images')
    parser.add_argument('--num_inference_steps', type=int, default=55,
                      help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=3.0,
                      help='Guidance scale for inference')
    parser.add_argument('--vae_scale_factor', type=int, default=8)
    return parser.parse_args()

def load_embedding_pipeline(model_directory, model_config, adapter_path, adapter_config=None):    
    pipeline = FluxPipeline.from_pretrained(
        model_directory,
        **model_config,
    )

    if adapter_path is None:
        return pipeline, None, None
    
    if adapter_path:
        adapter, network_alphas = pipeline.lora_state_dict(
            adapter_path,
            **adapter_config,
        )
        apply_lora_text_encoder(pipeline, adapter)
    return pipeline, adapter, network_alphas


def collect_embeddings(pipeline: FluxPipeline, prompt_directory:Path, input_formats):
    embeddings = []
    
    with torch.no_grad():
        for filepath in prompt_directory.iterdir():
            
            with open(filepath, "r") as file:
                prompt_data = json.loads(file.read())
                
                if "description" in prompt_data:
                    prompt_embeds, pooled_prompt_embeds, _ = pipeline.encode_prompt(
                        prompt=prompt_data["description"], prompt_2=None, max_sequence_length=512,
                    )
                    embed_device = prompt_embeds.device
                    embeddings.append((prompt_embeds, pooled_prompt_embeds, str(filepath), embed_device))
                else:
                    prompt_embeds, pooled_prompt_embeds, _ = pipeline.encode_prompt(
                        prompt=prompt_data['clip_l'], prompt_2=prompt_data['T5'], max_sequence_length=512,
                    )
                    embed_device = prompt_embeds.device
                    embeddings.append((prompt_embeds, pooled_prompt_embeds, str(filepath), embed_device))

    del pipeline.text_encoder
    del pipeline.text_encoder_2
    del pipeline.tokenizer
    del pipeline.tokenizer_2
    del pipeline

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    
    return embeddings


def get_sharded_model(model_directory, model_config):
    sharded_transformer = FluxTransformer2DModel.from_pretrained(
        model_directory,
        **model_config,
    )
    return sharded_transformer


def apply_lora_text_encoder(pipeline, state_dict):
    text_encoder_state_dict = {k: v for k, v in state_dict.items() if "text_encoder." in k}
    if len(text_encoder_state_dict) > 0:
        pipeline.load_lora_into_text_encoder(
            text_encoder_state_dict,
            network_alphas=None,
            text_encoder=pipeline.text_encoder,
            prefix="text_encoder",
            _pipeline=pipeline,
        )

    text_encoder_2_state_dict = {k: v for k, v in state_dict.items() if "text_encoder_2." in k}
    if len(text_encoder_2_state_dict) > 0:
        pipeline.load_lora_into_text_encoder(
            text_encoder_2_state_dict,
            network_alphas=None,
            text_encoder=pipeline.text_encoder_2,
            prefix="text_encoder_2",
            _pipeline=pipeline
        )

def apply_lora_transformer(transformer, state_dict):
    # Apply adapter to transformer
    transformer.load_lora_adapter(
        state_dict,        
        use_safetensors=True,
    )

def run_inference(
        model_directory, 
        text_encoder_config, 
        model_config, 
        scale_factor,
        prompt_directory,
        adapter_path=None, 
        adapter_options=None,
        height=768,
        width=768,
        num_inference_steps=55,
        guidance_scale=3.0,
        output_dir="test_batch",
        **kwargs):
    
    embedding_pipeline, adapter, _ = load_embedding_pipeline(
        model_directory,
        text_encoder_config,
        adapter_path,
        adapter_options,
    )

    transformer_lora_state_dict = None

    if adapter:
        # weights from adapter that will be applied to the transformer
        transformer_lora_state_dict = {
            k: adapter.pop(k) for k in list(adapter.keys()) if "transformer." in k and "lora" in k
        }

    embeddings = collect_embeddings(
        embedding_pipeline, 
        Path(prompt_directory),
        input_formats={".json", ".txt"}
    )

    sharded_transformer = get_sharded_model(model_directory, model_config)
    
    if adapter_path is not None:
        apply_lora_transformer(sharded_transformer, transformer_lora_state_dict)
    
    pipeline = FluxPipeline.from_pretrained(
        model_directory,
        text_encoder = None,
        text_encoder_2 = None,
        tokenizer = None,
        tokenizer_2 = None,
        vae = None,
        transformer = sharded_transformer,
        torch_dtype= torch.bfloat16,
        local_files_only=True,
    )

    # weights are on two devices
    transformer_device = next(sharded_transformer.parameters()).device

    vae = AutoencoderKL.from_pretrained(
        model_directory, 
        subfolder="vae", 
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )

    # do the inferencing for all of the prompts
    for prompt_embeds, pooled_prompt_embeds, prompt_path, embed_device in embeddings:
        prompt_name = prompt_path.split("/")[-1]

        if embed_device != transformer_device:
            # move to the same device
            prompt_embeds = prompt_embeds.to(transformer_device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(transformer_device)
        
        latents = pipeline(
            prompt_embeds = prompt_embeds,
            pooled_prompt_embeds = pooled_prompt_embeds,
            num_inference_steps = num_inference_steps,
            guidance_scale = guidance_scale,
            height=height,
            width=width,
            output_type = "latent",
        ).images

        latents_device = latents.device

        # processing can be done in a separate step
        vae.to(latents_device)
        image_processor = VaeImageProcessor(vae_scale_factor=scale_factor)

        with torch.no_grad():
            latents = FluxPipeline._unpack_latents(latents, height, width, scale_factor)
            latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor
            image = vae.decode(latents, return_dict=False)[0]
            image = image_processor.postprocess(image, output_type="pil")
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            name = f"{output_dir}/gen_no_caption_{prompt_name}-{timestamp}.png"
            print(name)
            image[0].save(name)


if __name__ == "__main__":
    args = parse_args()
    # assuming the adapter and model is downloaded locally, you can remove local_files_only=True but I've only tested it locally.
    
    # Create output directory if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    
    adapter_options = {
        "weight_name": args.adapter_weight_name,
        "return_alphas": True,
        "local_files_only": True,
        "use_safetensors": True,
    }

    # Device memory configuration for both GPUs if available
    max_memory = {0: args.gpu_memory}
    if torch.cuda.device_count() > 1:
        max_memory[1] = args.gpu_memory

    run_inference(
        args.model_path,
        {
            "vae": None,
            "transformer": None,
            "device_map": "balanced",
            "max_memory": max_memory,
            "torch_dtype": torch.bfloat16,
            "use_safetensor": True,
            "local_files_only": True,
        },
        {
            "subfolder": "transformer",
            "device_map": "auto",
            "max_memory": max_memory,
            "torch_dtype": torch.bfloat16,
            "local_files_only": True,
        },
        args.vae_scale_factor,
        args.prompt_directory,
        adapter_path=args.adapter_path,
        adapter_options=adapter_options,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        output_dir=args.output_dir
    )
