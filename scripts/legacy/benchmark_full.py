"""Comprehensive End-to-End Benchmark for AI Privacy Shield.

Runs multi-model protection and evaluates against:
- Multiple editors (InstructPix2Pix, Inpainting, Image2Image)
- Multiple tasks
- Multiple prompts per task
- Multiple seeds
- Multiple masks (for inpainting)
- Multiple transformations (JPEG, resize, etc.)
- Task-specific editing success metrics
"""
import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
import traceback

import torch
import numpy as np
from PIL import Image

BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND))

from app.editing.protector import apply_editing_protection
from app.editing.manager import get_editing_manager
from app.editing.adapter import EditRequest
from app.quality.metrics import compute_quality
from app.utils.imaging import load_and_normalize

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42]
OUT_DIR = Path("benchmark_results")

# Editors
EDITORS = {
    "instruct": {"repo": "timbrooks/instruct-pix2pix", "type": "instruction", "role": "evaluation"},
    "inpainting": {"repo": "runwayml/stable-diffusion-inpainting", "type": "inpainting", "role": "evaluation"},
    "image2image": {"repo": "stable-diffusion-v1-5/stable-diffusion-v1-5", "type": "image2image", "role": "held_out"},
}

# Tasks
TASKS = {
    "shirt_color": {
        "prompts": [
            "Change the shirt to red."
        ],
        "target": "a person wearing a red shirt",
        "mask_type": "clothing"
    },
    "hat": {
        "prompts": [
            "Add a hat."
        ],
        "target": "a person wearing a hat",
        "mask_type": "face"
    },
    "pencil_sketch": {
        "prompts": [
            "Convert the image into a pencil sketch."
        ],
        "target": "a pencil sketch drawing",
        "mask_type": "full"
    }
}

# Transformations
TRANSFORMATIONS = ["none", "resize"]

def _resize_to(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if arr.shape[:2] == size:
        return arr
    return np.asarray(Image.fromarray(arr).resize((size[1], size[0]), Image.LANCZOS))

def apply_transformation(img: np.ndarray, t_name: str) -> np.ndarray:
    if t_name == "none":
        return img.copy()
    elif t_name == "jpeg":
        import cv2
        _, enc = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 50])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1]
    elif t_name == "resize":
        import cv2
        h, w = img.shape[:2]
        small = cv2.resize(img, (int(w*0.8), int(h*0.8)), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return img

def get_mask(img_shape, mask_type):
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    if mask_type == "full":
        mask[...] = 1.0
    elif mask_type == "face":
        mask[h//6:h//2, w//3:2*w//3] = 1.0
    elif mask_type == "clothing":
        mask[h//2:, w//4:3*w//4] = 1.0
    return mask

def score_edit(clip, processor, img_orig: np.ndarray, img_out: np.ndarray, target: str) -> float:
    def _emb(img: np.ndarray, text: str):
        pil = Image.fromarray(img).convert("RGB")
        with torch.no_grad():
            inputs = processor(text=[text], images=pil, return_tensors="pt", padding=True).to(DEVICE)
            out = clip(**inputs)
        return out.image_embeds.detach().cpu().numpy().reshape(-1), out.text_embeds.detach().cpu().numpy().reshape(-1)

    def _cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    img_emb_orig, txt_emb = _emb(img_orig, target)
    img_emb_out, _ = _emb(img_out, target)
    return _cos(img_emb_out, txt_emb) - _cos(img_emb_orig, txt_emb)

def main():
    print("Starting Comprehensive AI-Editing Benchmark")
    img_path = Path("tests/fixtures/einstein.jpg")
    record = load_and_normalize(img_path.read_bytes(), len(img_path.read_bytes()))
    original_rgb = record.array

    OUT_DIR.mkdir(exist_ok=True, parents=True)
    Image.fromarray(original_rgb).save(OUT_DIR / "original.png")

    manager = get_editing_manager(DEVICE)
    
    # 1. Apply Protection
    print("Applying Protection...")
    t0 = time.time()
    edit_protect = apply_editing_protection(original_rgb, DEVICE, progress=lambda _i: None)
    protected_rgb = edit_protect.protected if edit_protect.applied else original_rgb.copy()
    print(f"Protection took {time.time() - t0:.1f}s. Applied: {edit_protect.applied}")
    
    Image.fromarray(protected_rgb).save(OUT_DIR / "protected.png")
    diff = np.abs(original_rgb.astype(np.int16) - protected_rgb.astype(np.int16)) * 5
    Image.fromarray(np.clip(diff, 0, 255).astype(np.uint8)).save(OUT_DIR / "difference.png")
    
    quality = compute_quality(original_rgb, protected_rgb, device=DEVICE)
    print(f"SSIM: {quality.ssim:.3f}, PSNR: {quality.psnr_db:.1f} dB")

    results = []

    # 2. Run Edits
    clip = manager.get_clip()
    processor = manager.clip_processor

    for editor_key, editor_info in EDITORS.items():
        print(f"\n--- Loading Editor: {editor_key} ---")
        try:
            editor = manager.get_editor(editor_info["repo"], editor_info["type"])
        except Exception as e:
            print(f"Failed to load {editor_key}: {e}")
            traceback.print_exc()
            continue

        for task_id, task_data in TASKS.items():
            print(f"  Task: {task_id}")
            mask = get_mask(original_rgb.shape, task_data["mask_type"])
            
            for prompt in task_data["prompts"]:
                for seed in SEEDS:
                    for transform in TRANSFORMATIONS:
                        # Prepare inputs
                        orig_t = apply_transformation(original_rgb, transform)
                        prot_t = apply_transformation(protected_rgb, transform)
                        
                        req_orig = EditRequest(
                            image_rgb=orig_t,
                            instruction=prompt,
                            seed=seed,
                            resolution=512,
                            num_inference_steps=20,
                            guidance_scale=7.5,
                            image_guidance_scale=1.5,
                            mask=mask,
                            strength=0.7
                        )
                        req_prot = EditRequest(
                            image_rgb=prot_t,
                            instruction=prompt,
                            seed=seed,
                            resolution=512,
                            num_inference_steps=20,
                            guidance_scale=7.5,
                            image_guidance_scale=1.5,
                            mask=mask,
                            strength=0.7
                        )
                        
                        try:
                            out_orig = editor.edit(req_orig)
                            out_prot = editor.edit(req_prot)
                            
                            score_orig = score_edit(clip, processor, orig_t, out_orig, task_data["target"])
                            score_prot = score_edit(clip, processor, prot_t, out_prot, task_data["target"])
                            
                            results.append({
                                "editor": editor_key,
                                "task": task_id,
                                "prompt": prompt,
                                "seed": seed,
                                "transform": transform,
                                "score_orig": score_orig,
                                "score_prot": score_prot,
                                "diff": score_orig - score_prot
                            })
                            print(f"    [{transform}] {prompt[:15]}... | Orig: {score_orig:.3f} | Prot: {score_prot:.3f}")
                        except Exception as e:
                            print(f"    Error on edit: {e}")
                            traceback.print_exc()
                            
    # Save results
    report = {
        "hardware": DEVICE,
        "protection": edit_protect.as_dict(),
        "quality": {
            "ssim": quality.ssim,
            "psnr": quality.psnr_db
        },
        "results": results
    }
    
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(report, f, indent=2)

    # Generate Markdown
    md = [
        "# AI PRIVACY SHIELD — AI EDITING PROTECTION BENCHMARK\n",
        "## Test Environment",
        f"- **Device:** {DEVICE}",
        f"- **Image:** einstein.jpg\n",
        "## Protection Configuration",
        f"- **Applied:** {edit_protect.applied}",
        f"- **SSIM:** {quality.ssim:.3f}",
        f"- **PSNR:** {quality.psnr_db:.1f} dB",
        f"- **Epsilon:** {edit_protect.epsilon:.3f}\n",
        "## Main Results\n",
        "| Editor | Task | Transform | Seed | Prompt | Orig Success | Prot Success | Absolute Δ |",
        "|---|---|---|---|---|---|---|---|"
    ]
    
    for r in results:
        p_short = r['prompt'][:20] + "..." if len(r['prompt']) > 20 else r['prompt']
        md.append(f"| {r['editor']} | {r['task']} | {r['transform']} | {r['seed']} | {p_short} | {r['score_orig']:.3f} | {r['score_prot']:.3f} | {r['diff']:.3f} |")
        
    with open(OUT_DIR / "report.md", "w") as f:
        f.write("\n".join(md))

    print(f"\nDone! Report saved to {OUT_DIR / 'report.md'}")

if __name__ == "__main__":
    main()
