# AI PRIVACY SHIELD — MULTI-FAMILY PROTECTION BENCHMARK

## Environment

- Device: **NVIDIA GeForce RTX 3050 A Laptop GPU** (cuda), 4093 MB VRAM
- Python 3.12.0 · torch 2.2.2+cu121 · diffusers 0.29.2 · transformers 4.41.2
- Red-team rounds: 2 · Seeds: [42] · Masks: ['shirt', 'hair', 'background', 'person', 'irregular'] · Transforms: ['jpeg_compression', 'resize']

## bohr.jpg

- Protection: applied · families: diffusion_editing, instruction_editing, inpainting, image_to_image, identity_reference, face_swap, vision_encoder · denoising error +12.7% · SSIM 0.990 · PSNR 39.8 dB
- Identity similarity (protected vs original): 0.9975 → 0.9911

### Adaptive red-team rounds

| Round | diffusion↑ | identity sim | vision sim | weakest | weights |
| --- | ---: | ---: | ---: | --- | --- |
| 1 | 0.159 | 0.995 | 0.829 | identity | d1.00/i0.60/v0.35 |
| 2 | 0.158 | 0.995 | 0.830 | identity | d1.00/i0.90/v0.35 |

### Direct editing (original vs protected)

| Editor | Task | Mask | Orig | Prot | Abs Δ | Rel Δ |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| instruction | Change shirt color | — | 0.527 | 0.464 | +0.063 | 11.9% |
| instruction | Change background | — | 0.492 | 0.423 | +0.070 | 14.2% |
| instruction | Add a hat | — | 0.699 | 0.523 | +0.176 | 25.2% |
| instruction | Convert to pencil sketch | — | 0.060 | 0.049 | +0.011 | 19.2% |
| inpainting | Change shirt color | shirt | 0.008 | 0.006 | +0.002 | n/a |
| inpainting | Change background | background | 0.857 | 0.820 | +0.037 | 4.4% |
| inpainting | Add a hat | hair | 0.307 | 0.325 | -0.018 | -5.9% |
| inpainting | Convert to pencil sketch | person | 0.000 | 0.000 | +0.000 | n/a |
| image2image | Change background | — | 0.331 | 0.351 | -0.019 | -5.9% |
| image2image | Convert to pencil sketch | — | 0.272 | 0.309 | -0.037 | -13.7% |

- **Direct-editing aggregate:** 0.355 → 0.327 (abs +0.029); 6/10 rows reduced

### Edit success after transformations

| Transform | Orig | After |
| --- | ---: | ---: |
| jpeg_compression | 0.510 | 0.474 |
| resize | 0.510 | 0.433 |

### Identity-reference / face-swap (E-F)

| Model | Role | Sim orig→prot | Disruption |
| --- | --- | ---: | ---: |
| FaceNet (CASIA-WebFace) — evaluation | evaluation | 0.9918 | 0.0082 |
| ArcFace (MobileFaceNet) — held out | held_out | 0.9954 | 0.0046 |

### Vision encoders (I) + VLM conditioning (H)

| Model | Role | Sim orig→prot | Disruption |
| --- | --- | ---: | ---: |
| MobileNetV3-Large — optimization | optimization | 0.8301 | 0.1699 |
| ResNet50 — held out | held_out | 0.931 | 0.069 |
| CLIP ViT-L/14 — evaluation | evaluation | 0.9766 | 0.0234 |

### Image-to-video (G)

- NOT TESTED: modular adapter registered; local video-generation weights exceed this hardware. Identity disruption (E/F) is the surrogate evidence.

## curie.jpg

- Protection: applied · families: diffusion_editing, instruction_editing, inpainting, image_to_image, identity_reference, face_swap, vision_encoder · denoising error +0.0% · SSIM 0.994 · PSNR 38.4 dB
- Identity similarity (protected vs original): 0.9909 → 0.9786

### Adaptive red-team rounds

| Round | diffusion↑ | identity sim | vision sim | weakest | weights |
| --- | ---: | ---: | ---: | --- | --- |
| 1 | -0.025 | 0.990 | 0.510 | diffusion | d1.00/i0.60/v0.35 |
| 2 | -0.066 | 0.989 | 0.347 | diffusion | d1.50/i0.60/v0.35 |

### Direct editing (original vs protected)

| Editor | Task | Mask | Orig | Prot | Abs Δ | Rel Δ |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| instruction | Change shirt color | — | 0.466 | 0.216 | +0.250 | 53.7% |
| instruction | Change background | — | 0.499 | 0.633 | -0.135 | -27.0% |
| instruction | Add a hat | — | 0.581 | 0.469 | +0.112 | 19.3% |
| instruction | Convert to pencil sketch | — | 0.116 | 0.031 | +0.085 | 73.0% |
| inpainting | Change shirt color | shirt | 0.029 | 0.009 | +0.020 | 68.1% |
| inpainting | Change background | background | 0.887 | 0.868 | +0.019 | 2.2% |
| inpainting | Add a hat | hair | 0.777 | 0.802 | -0.025 | -3.2% |
| inpainting | Convert to pencil sketch | person | 0.006 | 0.000 | +0.006 | n/a |
| image2image | Change background | — | 0.329 | 0.313 | +0.016 | 4.9% |
| image2image | Convert to pencil sketch | — | 0.399 | 0.277 | +0.123 | 30.8% |

- **Direct-editing aggregate:** 0.409 → 0.362 (abs +0.047); 8/10 rows reduced

### Edit success after transformations

| Transform | Orig | After |
| --- | ---: | ---: |
| jpeg_compression | 0.482 | 0.429 |
| resize | 0.482 | 0.407 |

### Identity-reference / face-swap (E-F)

| Model | Role | Sim orig→prot | Disruption |
| --- | --- | ---: | ---: |
| FaceNet (CASIA-WebFace) — evaluation | evaluation | 0.9874 | 0.0126 |
| ArcFace (MobileFaceNet) — held out | held_out | 0.9688 | 0.0312 |

### Vision encoders (I) + VLM conditioning (H)

| Model | Role | Sim orig→prot | Disruption |
| --- | --- | ---: | ---: |
| MobileNetV3-Large — optimization | optimization | 0.3474 | 0.6526 |
| ResNet50 — held out | held_out | 0.7617 | 0.2383 |
| CLIP ViT-L/14 — evaluation | evaluation | 0.9331 | 0.0669 |

### Image-to-video (G)

- NOT TESTED: modular adapter registered; local video-generation weights exceed this hardware. Identity disruption (E/F) is the surrogate evidence.

## lincoln.jpg

- Protection: applied · families: diffusion_editing, instruction_editing, inpainting, image_to_image, identity_reference, face_swap, vision_encoder · denoising error +13.0% · SSIM 0.997 · PSNR 39.8 dB
- Identity similarity (protected vs original): 0.9963 → 0.9944

### Adaptive red-team rounds

| Round | diffusion↑ | identity sim | vision sim | weakest | weights |
| --- | ---: | ---: | ---: | --- | --- |
| 1 | 0.165 | 0.997 | 0.736 | identity | d1.00/i0.60/v0.35 |
| 2 | 0.168 | 0.997 | 0.738 | identity | d1.00/i0.90/v0.35 |

### Direct editing (original vs protected)

| Editor | Task | Mask | Orig | Prot | Abs Δ | Rel Δ |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| instruction | Change shirt color | — | 0.460 | 0.378 | +0.082 | 17.8% |
| instruction | Change background | — | 0.285 | 0.257 | +0.028 | 9.8% |
| instruction | Add a hat | — | 0.734 | 0.733 | +0.002 | 0.2% |
| instruction | Convert to pencil sketch | — | 0.040 | 0.036 | +0.004 | 11.0% |
| inpainting | Change shirt color | shirt | 0.035 | 0.000 | +0.035 | 99.0% |
| inpainting | Change background | background | 0.953 | 0.934 | +0.019 | 2.0% |
| inpainting | Add a hat | hair | 0.622 | 0.687 | -0.065 | -10.5% |
| inpainting | Convert to pencil sketch | person | 0.025 | 0.000 | +0.025 | 100.0% |
| image2image | Change background | — | 0.479 | 0.546 | -0.068 | -14.1% |
| image2image | Convert to pencil sketch | — | 0.350 | 0.319 | +0.032 | 9.0% |

- **Direct-editing aggregate:** 0.399 → 0.389 (abs +0.009); 8/10 rows reduced

### Edit success after transformations

| Transform | Orig | After |
| --- | ---: | ---: |
| jpeg_compression | 0.373 | 0.315 |
| resize | 0.373 | 0.312 |

### Identity-reference / face-swap (E-F)

| Model | Role | Sim orig→prot | Disruption |
| --- | --- | ---: | ---: |
| FaceNet (CASIA-WebFace) — evaluation | evaluation | 0.9973 | 0.0027 |
| ArcFace (MobileFaceNet) — held out | held_out | 0.9958 | 0.0042 |

### Vision encoders (I) + VLM conditioning (H)

| Model | Role | Sim orig→prot | Disruption |
| --- | --- | ---: | ---: |
| MobileNetV3-Large — optimization | optimization | 0.7381 | 0.2619 |
| ResNet50 — held out | held_out | 0.9305 | 0.0695 |
| CLIP ViT-L/14 — evaluation | evaluation | 0.9873 | 0.0127 |

### Image-to-video (G)

- NOT TESTED: modular adapter registered; local video-generation weights exceed this hardware. Identity disruption (E/F) is the surrogate evidence.

## tesla.jpg

- Protection: applied · families: diffusion_editing, instruction_editing, inpainting, image_to_image, identity_reference, face_swap, vision_encoder · denoising error +0.0% · SSIM 0.995 · PSNR 39.0 dB
- Identity similarity (protected vs original): 0.99 → 0.9675

### Adaptive red-team rounds

| Round | diffusion↑ | identity sim | vision sim | weakest | weights |
| --- | ---: | ---: | ---: | --- | --- |
| 1 | -0.027 | 0.980 | 0.735 | diffusion | d1.00/i0.60/v0.35 |
| 2 | -0.046 | 0.969 | 0.726 | diffusion | d1.50/i0.60/v0.35 |

### Direct editing (original vs protected)

| Editor | Task | Mask | Orig | Prot | Abs Δ | Rel Δ |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| instruction | Change shirt color | — | 0.917 | 0.795 | +0.123 | 13.4% |
| instruction | Change background | — | 0.429 | 0.846 | -0.416 | -97.0% |
| instruction | Add a hat | — | 0.208 | 0.226 | -0.018 | -8.6% |
| instruction | Convert to pencil sketch | — | 0.021 | 0.046 | -0.024 | -112.8% |
| inpainting | Change shirt color | shirt | 0.041 | 0.010 | +0.031 | 75.9% |
| inpainting | Change background | background | 0.837 | 0.803 | +0.034 | 4.0% |
| inpainting | Add a hat | hair | 0.424 | 0.297 | +0.127 | 30.0% |
| inpainting | Convert to pencil sketch | person | 0.004 | 0.000 | +0.004 | n/a |
| image2image | Change background | — | 0.357 | 0.473 | -0.117 | -32.7% |
| image2image | Convert to pencil sketch | — | 0.400 | 0.355 | +0.045 | 11.3% |

- **Direct-editing aggregate:** 0.364 → 0.385 (abs -0.021); 6/10 rows reduced

### Edit success after transformations

| Transform | Orig | After |
| --- | ---: | ---: |
| jpeg_compression | 0.673 | 0.845 |
| resize | 0.673 | 0.791 |

### Identity-reference / face-swap (E-F)

| Model | Role | Sim orig→prot | Disruption |
| --- | --- | ---: | ---: |
| FaceNet (CASIA-WebFace) — evaluation | evaluation | 0.9706 | 0.0294 |
| ArcFace (MobileFaceNet) — held out | held_out | 0.9484 | 0.0516 |

### Vision encoders (I) + VLM conditioning (H)

| Model | Role | Sim orig→prot | Disruption |
| --- | --- | ---: | ---: |
| MobileNetV3-Large — optimization | optimization | 0.726 | 0.274 |
| ResNet50 — held out | held_out | 0.8789 | 0.1211 |
| CLIP ViT-L/14 — evaluation | evaluation | 0.9722 | 0.0278 |

### Image-to-video (G)

- NOT TESTED: modular adapter registered; local video-generation weights exceed this hardware. Identity disruption (E/F) is the surrogate evidence.

## einstein.jpg

- Protection: applied · families: diffusion_editing, instruction_editing, inpainting, image_to_image, identity_reference, face_swap, vision_encoder · denoising error +7.4% · SSIM 0.980 · PSNR 40.0 dB
- Identity similarity (protected vs original): 0.9959 → 0.995

### Adaptive red-team rounds

| Round | diffusion↑ | identity sim | vision sim | weakest | weights |
| --- | ---: | ---: | ---: | --- | --- |
| 1 | 0.071 | 0.998 | 0.816 | identity | d1.00/i0.60/v0.35 |
| 2 | 0.080 | 0.998 | 0.819 | identity | d1.00/i0.90/v0.35 |

### Direct editing (original vs protected)

| Editor | Task | Mask | Orig | Prot | Abs Δ | Rel Δ |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| instruction | Change shirt color | — | 0.671 | 0.496 | +0.174 | 26.0% |
| instruction | Change background | — | 0.683 | 0.593 | +0.091 | 13.3% |
| instruction | Add a hat | — | 0.594 | 0.566 | +0.028 | 4.7% |
| instruction | Convert to pencil sketch | — | 0.088 | 0.039 | +0.049 | 55.3% |
| inpainting | Change shirt color | shirt | 0.052 | 0.011 | +0.041 | 79.5% |
| inpainting | Change background | background | 0.923 | 0.833 | +0.090 | 9.8% |
| inpainting | Add a hat | hair | 0.475 | 0.469 | +0.006 | 1.3% |
| inpainting | Convert to pencil sketch | person | 0.020 | 0.005 | +0.015 | n/a |
| image2image | Change background | — | 0.607 | 0.530 | +0.076 | 12.5% |
| image2image | Convert to pencil sketch | — | 0.267 | 0.079 | +0.189 | 70.6% |

- **Direct-editing aggregate:** 0.438 → 0.362 (abs +0.076); 10/10 rows reduced

### Edit success after transformations

| Transform | Orig | After |
| --- | ---: | ---: |
| jpeg_compression | 0.677 | 0.599 |
| resize | 0.677 | 0.479 |

### Identity-reference / face-swap (E-F)

| Model | Role | Sim orig→prot | Disruption |
| --- | --- | ---: | ---: |
| FaceNet (CASIA-WebFace) — evaluation | evaluation | 0.9939 | 0.0061 |
| ArcFace (MobileFaceNet) — held out | held_out | 0.9905 | 0.0095 |

### Vision encoders (I) + VLM conditioning (H)

| Model | Role | Sim orig→prot | Disruption |
| --- | --- | ---: | ---: |
| MobileNetV3-Large — optimization | optimization | 0.8187 | 0.1813 |
| ResNet50 — held out | held_out | 0.9442 | 0.0558 |
| CLIP ViT-L/14 — evaluation | evaluation | 0.9722 | 0.0278 |

### Image-to-video (G)

- NOT TESTED: modular adapter registered; local video-generation weights exceed this hardware. Identity disruption (E/F) is the surrogate evidence.
