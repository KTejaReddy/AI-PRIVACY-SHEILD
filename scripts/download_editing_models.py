"""Download local editing models for AI Privacy Shield.

This downloads the heavy models needed for the editing-protection benchmark:

  * instruct   — InstructPix2Pix (held-out real editing benchmark)
  * surrogate  — SD1.5 base (differentiable anti-diffusion optimizer + img2img editor)
  * inpainting — SD1.5 inpainting U-Net (masked-inpainting attack; VAE/text
                 encoder/tokenizer/scheduler are shared with the SD1.5 base)
  * clip       — CLIP ViT-L/14 (auxiliary semantic scorer)

Uses hf_transfer for fast downloads. Everything stays in the local Hugging Face
cache (~11 GB total); nothing is uploaded anywhere.
"""

import argparse
import os
import sys

from huggingface_hub import hf_hub_download, snapshot_download

MODELS = {
    "instruct": "timbrooks/instruct-pix2pix",
    "surrogate": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "inpainting": "stable-diffusion-v1-5/stable-diffusion-inpainting",
    "clip": "openai/clip-vit-large-patch14",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="all", help="comma-separated model keys or 'all'")
    args = parser.parse_args()

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    keys = list(MODELS.keys()) if args.models == "all" else args.models.split(",")

    for key in keys:
        if key not in MODELS:
            print(f"Unknown model key: {key}")
            continue
        repo = MODELS[key]
        print(f"Downloading {repo}...")
        if key == "inpainting":
            # Only the 9-channel U-Net differs from SD1.5; grab the fp16 variant.
            for f in ("config.json", "diffusion_pytorch_model.fp16.safetensors"):
                p = hf_hub_download(repo_id=repo, filename=f"unet/{f}")
                print(f"  ok unet/{f} ({os.path.getsize(p) / 2**20:.0f} MB)")
        else:
            snapshot_download(
                repo_id=repo,
                allow_patterns=["*.fp16.safetensors", "*.safetensors", "*.json", "*.txt"],
                ignore_patterns=["*.ckpt", "*.pt", "*.pth", "*.msgpack", "*.h5", "*non_ema*"],
            )
    print("Done. All editing models are local.")


if __name__ == "__main__":
    sys.exit(main())
