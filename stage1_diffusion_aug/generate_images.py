"""
Stage 1 — TEL Image Generation via Textual Inversion

Generates diverse synthetic TEL (Transporter-Erector-Launcher) images on
white backgrounds using a Stable Diffusion pipeline whose text encoder has
been extended with a fine-tuned Textual Inversion embedding.

Workflow:
  1. Load a pre-trained SD model (CompVis/stable-diffusion-v1-4).
  2. Inject the TEL-specific token embedding (learned by fine_tune.py) into
     the tokenizer and text encoder.
  3. Optionally load an ESD (Erased Stable Diffusion) UNet checkpoint to
     suppress unwanted concepts.
  4. Run unconditional text-to-image generation with the custom TEL token and
     save each output as a numbered PNG on a white background.

Usage:
  python generate_images.py \
      --embed-path tel-tokens/<class>/learned_embeds.bin \
      --prompt "a photo of a <tel_vehicle>, white background" \
      --num-generate 50 \
      --out output/tel_class/
"""

from semantic_aug.augmentations.textual_inversion import TextualInversion
from diffusers import StableDiffusionPipeline
from torch import autocast
from PIL import Image

from tqdm import trange
import os
import torch
import argparse
import numpy as np
import random


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Stage 1: TEL image generation via Textual Inversion")

    # Stable Diffusion base model (HuggingFace Hub path or local directory)
    parser.add_argument("--model-path", type=str, default="CompVis/stable-diffusion-v1-4")
    # Path to the learned_embeds.bin produced by fine_tune.py
    parser.add_argument("--embed-path", type=str, required=True,
                        help="Path to the Textual Inversion learned_embeds.bin for the TEL class")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-generate", type=int, default=50,
                        help="Number of synthetic TEL images to generate")

    # The custom token (e.g. <buk_m2>) must match the placeholder used in fine_tune.py
    parser.add_argument("--prompt", type=str, default="a photo of a <tel_vehicle>, white background")
    parser.add_argument("--out", type=str, default="output/generated/",
                        help="Directory to save generated PNG images")

    parser.add_argument("--guidance-scale", type=float, default=7.5)
    # Optional: path to an ESD-modified UNet checkpoint that suppresses irrelevant concepts
    parser.add_argument("--erasure-ckpt-name", type=str, default=None,
                        help="(Optional) ESD UNet checkpoint path for concept suppression")

    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load the base SD pipeline in fp16 for memory efficiency
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_path, use_auth_token=True,
        revision="fp16",
        torch_dtype=torch.float16
    ).to('cuda')

    # Replace the tokenizer & text encoder with TEL-token-extended versions
    aug = TextualInversion(args.embed_path, model_path=args.model_path)
    pipe.tokenizer = aug.pipe.tokenizer
    pipe.text_encoder = aug.pipe.text_encoder

    pipe.set_progress_bar_config(disable=True)
    pipe.safety_checker = None  # disable for domain-specific military imagery

    # Optionally apply an ESD-modified UNet to suppress background concepts
    if args.erasure_ckpt_name is not None:
        pipe.unet.load_state_dict(torch.load(args.erasure_ckpt_name, map_location='cuda'))

    # Generate and save each synthetic TEL image
    for idx in trange(args.num_generate, desc="Generating TEL images"):
        with autocast('cuda'):
            image = pipe(
                args.prompt,
                guidance_scale=args.guidance_scale
            ).images[0]
        image.save(os.path.join(args.out, f"{idx}.png"))