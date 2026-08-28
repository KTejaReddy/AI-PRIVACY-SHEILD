# Limitations and threat model

## Honest positioning

AI Privacy Shield provides **maximum practical privacy against defined and
tested threats**. It does not provide "AI-proof" images, and it never claims
to. Protection is evaluated against the surrogate models installed locally, a
fixed set of transformations, and the tested editing pipeline; a different
recognition system, a different editor, a tailored attack, or a stronger
perturbation budget than the quality constraints allow can defeat it.

## Threat model (what this defends against)

- Automated identity matching against recognition models **similar in
  architecture/training to the tested surrogates** (FaceNet VGGFace2 / CASIA,
  ArcFace w600k).
- Face detection by the tested detectors: the OpenCV DNN SSD and the MTCNN
  cascade are both optimization and evaluation targets (differentiable
  MTCNN P/R/O-Net surrogates in phase 1; logit-space black-box suppression in
  phase 2).
- Person detection by the tested detectors: OpenCV HOG + SVM and a neural
  Faster R-CNN ResNet-50 (torchvision COCO weights). Person boxes widen the
  protection mask and both detectors are suppression objectives.
- General vision features: a MobileNetV3-Large ImageNet encoder is an
  optimization surrogate; a ResNet50 ImageNet encoder is a **held-out**
  evaluation model (never optimized against, so it tests transferability).
- Common post-processing: JPEG compression, resizing, mild cropping,
  brightness/contrast changes, re-encoding.
- Metadata-based tracking (GPS/EXIF) — fully removed.

## Measured behavior of the tested detectors (September 2026 build)

These are the real, measured outcomes on the bundled test portrait with the
9/255 perturbation bound and SSIM ≥ 0.9 floors — they differ per detector and
per image, and the report always shows the live measurements:

- **OpenCV SSD face detector: essentially invariant.** Even ±32/255
  structured noise changes its confidence by < 0.005; within the 9/255 budget
  it moves ~0.001. The res10 SSD is a genuinely hard target for imperceptible
  perturbation. The system reports the honest (small) change rather than a
  fake win.
- **MTCNN cascade: modestly suppressed** (typically ~0.97 → ~0.94–0.95).
  Its P/R/O-Net stages are the differentiable surrogate, so this is the face
  detector the attack actually moves; it does not usually cause a detection
  miss at the 9/255 budget.
- **HOG person detector: strongly suppressed** — when a full body is present,
  weight typically drops to zero. It misses portraits entirely (honestly
  reported).
- **Faster R-CNN person detector: meaningfully but variably suppressed**
  (measured −6% to −58% depending on trajectory).

## Multi-family protection limits (measured)

- The multi-family perturbation is optimized against the **SD1.5** denoising
  surrogate, the **FaceNet (VGGFace2)** identity encoder and the
  **MobileNetV3** vision encoder, and benchmarked against **InstructPix2Pix**
  (held out), **SD1.5 masked inpainting**, **SD1.5 image-to-image**, **FaceNet
  (CASIA)**, **ArcFace (held out)**, **ResNet50 (held out)** and **CLIP**. It
  is *not* claimed to transfer to every editor or generator.
- **Identity-reference / face-swap protection is surrogate evidence**: we
  disrupt the face-embedding representation that reference-conditioned
  generators and face-swap pipelines extract identity from, but no local
  reference-conditioned generator or face-swap model is run end to end on
  this hardware. Results are reported as embedding-similarity disruption, not
  as a defeated proprietary swap service.
- **Image-to-video (family G) is NOT TESTED** on this hardware: the modular
  adapter is registered in the attack registry, but local video-generation
  weights exceed the development GPU. The registry and the benchmark report
  it as not tested rather than fabricating a result.
- Identity disruption by an imperceptible perturbation is modest for
  FaceNet-class encoders (a few points of cosine similarity at the 4.5/255
  budget) — this is measured and reported as-is, never inflated.

## AI-editing protection limits (measured)

- The anti-diffusion perturbation is optimized against the **SD1.5** denoising
  surrogate and benchmarked against **InstructPix2Pix**, **SD1.5 masked
  inpainting** and **SD1.5 image-to-image**. It is *not* claimed to transfer
  to every editor — different architectures (GAN-based editors, editing models
  with their own VAEs/conditioners, commercial APIs with unknown internals)
  may be unaffected.
- Edit-success is a composite of a **task-specific pixel metric** (primary)
  and CLIP semantic alignment (auxiliary). Raw components are always reported;
  the relative percentage is omitted when the original success is too small to
  be meaningful, and absolute change is the headline number.
- The inpainting evaluation uses **real region masks** derived from the
  detected face/person boxes. Masks depend on detector quality: if no person
  is detected (HOG is full-body only), the shirt/background regions fall back
  to geometric bands, which may not match the true subject.
- The editing stage runs only when the local models are present; otherwise it
  reports "unavailable on this hardware" and the recognition layers still
  apply.
- The perturbation budget for the editing stage is intentionally smaller
  (≈ 4.5/255 vs 9/255) to keep the image visibly identical; stronger
  disruption requires accepting visible change.
- **Mask / prompt / seed dependency:** protection is measured over the masks,
  prompts and seeds in the benchmark. An attacker who picks an unseen mask,
  prompt phrasing, or seed may see different (better) edit success; the report
  states exactly which were tested.

## Measured robustness finding

Under the imperceptible-perturbation budget (ε ≈ 9/255, SSIM ≥ 0.9, PSNR
≥ 30 dB), embedding disruption that is strong on the clean image (FaceNet
cosine similarity −10…−12%) degrades under lossy post-processing: JPEG-70 /
resize / re-encode keep roughly half of it, which lands just below the
`DISRUPT_PARTIAL` (0.40) threshold for the face models. The robustness test
reports this honestly as FAIL/PARTIAL per transform. This is a real property
of imperceptible adversarial perturbations on FaceNet-class models, not a
threshold artifact; raising the budget above the quality floors is the main
lever if stronger post-processing survival is required.

## What is outside the threat model

- A recognition system trained on the user's *protected* images (adaptive
  attacker).
- Very large perturbation budgets that destroy visual similarity — the system
  bounds `‖δ‖∞ ≤ ε` to keep images natural, and that bound is a hard cap.
- Physical attacks (re-photographing a screen, 3D-printed faces).
- Human recognition (a person who knows you will still recognize you; this is
  anti-AI, not a disguise).
- OCR: text/PII detection is heuristic and explicitly experimental — it can
  miss or misread text.
- Face detection quality on very small, blurry, or heavily occluded faces.
- Person detection (HOG): full-body only — portraits, seated people, and
  occluded bodies are commonly missed. The perception report reflects what
  the detector actually found; missing a person is not reported as protection.
- VLM semantic interpretation: not enabled in this configuration (no local
  vision-language model is bundled). The UI states this explicitly rather
  than pretending semantic protection happened.

## Model coverage

The verification report lists exactly which models were tested. If ArcFace is
not installed (weights not downloaded), the verification and black-box
refinement phases use FaceNet only, and the report says so. Never assume a
model that isn't listed was tested.

## Performance

- GPU (CUDA) gives the full optimization budget (70 iterations, 48
  refinement steps). CPU mode automatically drops to 35 iterations / 16
  refinement steps with fewer queries — faster but weaker protection.
- First run downloads FaceNet weights into the torch cache; later runs load
  from disk.
- Large photos (e.g. 12 MP) are processed at native resolution where possible;
  very large dimensions are capped by `AIPS_MAX_IMAGE_DIMENSION`.

## Ethical and legal notes

- **Consent:** only protect faces and content you are entitled to process.
  Applying protection to a photo of someone else without their knowledge may
  be inappropriate or unlawful in your jurisdiction (GDPR/CCPA-style rules,
  consent requirements, "right to be forgotten" flows).
- **Anti-evasion framing:** this is a privacy tool for individuals protecting
  their own images, not a tool for evading lawful identification in contexts
  where identity verification is required (KYC, security checkpoints, court
  evidence).
- **No guarantee of anonymization:** the output may still contain identifying
  information (clothing, tattoos, context) that no face-protection system can
  remove.
