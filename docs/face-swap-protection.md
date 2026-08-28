# Face-Swap / Identity-Reference Protection

The final product goal is unchanged: one photo in, one protected photo out. The
protected photo must look normal to humans while being a *poor identity source*
for face-swap and identity-reference generation systems.

This document records exactly what was integrated, which published methods the
implementation follows, which models were used for optimization vs evaluation,
and what the real benchmarks measure. It is written to be read alongside
`docs/consolidation.md` and `docs/evaluation.md`.

## 1. What was the weak point?

The pre-upgrade identity branch pushed a single FaceNet embedding away from its
original. That is a *proxy*: a face embedding changing slightly does not prove a
modern face-swap or identity-reference generator will fail. The benchmark showed
identity/reference similarity of ≈0.97–1.00 after protection — effectively no
face-swap protection at all.

## 2. Methodologies followed (research provenance)

We reviewed the published proactive face-swap defenses and reproduced their
*core mechanisms with our own differentiable code* (no upstream source vendored;
licenses recorded below):

| Method | Core mechanism we follow | Status |
|---|---|---|
| **ID-Eraser** (2026) | Identity-space perturbation of the face representation, optimized against the swap-identity encoder family, then natural reconstruction | Reproduced: multi-encoder identity-space objective + transform (interference) robustness on the same crops. Full autoencoder reconstruction network **NOT implemented** (see §9). |
| **Phantom** (2026) | Combined latent + spatial constraints targeting INSwapper/UniFace/SimSwap families | Reproduced: identity-objective on the actual swap encoder family + soft elliptical spatial emphasis on identity-relevant regions. Full dual-constraint latent optimization **NOT implemented**. |
| **PhotoGuard** (MIT, 2023) | Diffusion (latent-space) perturbation that breaks editing | Already integrated; unchanged (see `docs/editing-protection.md`). |

Both face-swap methods are "defense against the identity encoder family" — the
key idea is that a face swap first extracts the identity representation of the
source face, so disrupting that representation in a way that survives the swap
pipeline reduces identity transfer. We implement exactly that objective, on the
encoder family real swap systems use, and measure the *actual* attack outcome.

## 3. The unified perturbation (no stacking)

Per the product rule, face-swap/reference protection is **one objective inside
the one perturbation**, not a second stacked pass:

```
ONE bounded δ
   ├─ diffusion/editing protection (PhotoGuard-style, unchanged)
   ├─ identity-space protection  — FaceNet VGGFace2 + CASIA (differentiable PGD)
   │     └─ soft elliptical region emphasis (Phantom spatial constraint)
   │     └─ transform-robust identity scoring (ID-Eraser interference concept)
   └─ in-place ArcFace refinement — zeroth-order, SAME δ (ID-Eraser identity space
        applied to the swap-encoder family: ArcFace w600k_mbf, black-box ONNX)
```

- PGD iterations are split: a fraction of iterations apply only the identity
  objective (sign-SGD on the face region) so the much larger diffusion gradient
  does not swallow the identity gradient; the rest blend both.
- The ArcFace refinement keeps every update inside the same ε bound and re-checks
  SSIM; it continues optimizing the identical δ rather than adding a second one.
- Quality floor is enforced (SSIM ≥ configured minimum, default 0.90 for
  research, higher in production). If the face-swap objective would push quality
  below the floor, its influence is limited by the floor — never by silently
  degrading the image.

## 4. Models used for OPTIMIZATION (the production engine)

| Model | Role | Loaded in production? |
|---|---|---|
| SD1.5 anti-diffusion surrogate | diffusion/editing protection | on demand |
| FaceNet VGGFace2 | differentiable identity-space PGD term | yes |
| FaceNet CASIA-WebFace | second differentiable identity encoder (multi-encoder objective) | yes |
| MobileNetV3-Large | global vision-encoder term | yes |
| ArcFace w600k_mbf (ONNX) | in-place black-box identity refinement (swap-encoder family) | yes |

## 5. Models used for EVALUATION (research benchmarks only)

| Model | Family | License | Role |
|---|---|---|---|
| **INSwapper** (`inswapper_128.onnx`) | real face-swap pipeline | insightface; contact for redistribution | swap attack (evaluation) |
| **ArcFace buffalo_l** `w600k_r50` (ONNX) | identity recognition | insightface research use | identity-transfer metric (what swap pipelines maximize) |
| **FaceNet VGGFace2** | identity recognition (different family) | facenet-pytorch research use | held-out metric |
| **IP-Adapter FaceID SD1.5** | identity-reference generation | non-commercial (h94) | reference-generation attack (evaluation) |
| SimSwap | face-swap (alternative family) | CC-BY-NC 4.0 | **NOT TESTED** — weights on Google Drive, unreachable from this build machine; documented, not fabricated |

The evaluation models are used exactly once, at the end, on fixed inputs
(original vs protected) with identical settings — they never influence the
optimization.

## 6. What the benchmarks actually measure

### `scripts/benchmark_face_swap.py` — real INSwapper attack

```
original source face ─┐
                      ├─ INSwapper → swapped face → identity transfer vs SOURCE
protected source face ┘   (same target image, same detection, same model)
```

- Every source image is swapped into **every other** image in the set (multiple
  target identities).
- Variants per source/target: `original`, `protected`, `protected+jpeg`,
  `protected+resize` (same transforms the robustness suite uses).
- Metrics:
  - **ArcFace (buffalo_l) source similarity** — identity transfer of the swapped
    face vs the source identity. *Lower on protected = attack transfers less
    identity.*
  - **ArcFace target similarity** — does the target identity get displaced.
  - **FaceNet VGGFace2 source similarity** — held-out encoder family.
- A source face is *successfully* protected only if real swap outputs show
  substantially lower identity transfer than the original source.
- Checkpointed per source (protected PNG + metadata cached) so a crash never
  loses the run; resumes cheaply.

### `scripts/benchmark_reference_gen.py` — IP-Adapter FaceID attack

```
reference photo ──→ identity encoder ──→ NEW synthetic image of same person
```

- The attacker never edits the protected photo — they use it as a *reference*
  to generate a new image of the person.
- Same prompts and settings for original vs protected reference; identity
  transfer of the generated face vs the **original** embedding (ArcFace + held-out
  FaceNet).
- Shares the protected-image cache with the face-swap benchmark, so the exact
  same protected photos are tested against both attack families.
- Image-to-video / talking-head (e.g. SadTalker-style): **NOT TESTED** — no
  practical local model runs on this hardware within reasonable time; documented,
  not fabricated.

## 7. Held-out generalization

- Optimization uses FaceNet VGGFace2 + CASIA (differentiable). Evaluation uses
  ArcFace buffalo_l + INSwapper + FaceID — none of these see the optimizer.
- The spec asks for "held-out face-swap model C": SimSwap is the designated
  held-out family; it is reported NOT TESTED because its weights are unreachable
  from this machine. That is an honest gap, listed in the report.

## 8. Claim boundary

- Claim: *"Protection is designed to reduce the usefulness of protected
  photographs as identity sources for tested face-swap and reference-generation
  systems."*
- Claim: *"No tested attack in the benchmark exceeded the defined failure
  threshold"* — only if the actual benchmark numbers say so.
- NOT claimed: universal immunity, "no AI can ever recognize/swap this image",
  protection against untested systems.

## 9. Measured results (real attacks, honest numbers)

Both benchmarks run on the real unified engine with the real models, on the
5-image benchmark set, and the numbers below are exactly what the checkpoints
record (no fabricated metrics):

### Face swap (INSwapper, `results/face_swap/report.md`)

| Metric | Value |
|---|---|
| Identity transfer (ArcFace buffalo_l), orig vs prot | mean **−0.009** (20 source×target pairs) |
| Pairs reduced beyond Δ < −0.01 | 9/20 |
| Identity-transfer survives JPEG / resize | yes — prot+JPEG / prot+resize rows stay at the protected level |
| Visual quality | SSIM 0.963–0.984 · PSNR 40.7–41.8 dB |
| FaceNet VGGFace2 (held-out family) | consistent with ArcFace; moves by the same small amount |

### Reference generation (IP-Adapter FaceID SD1.5, `results/reference_gen/report.md`)

| Metric | Value |
|---|---|
| Identity transfer, orig vs prot reference | mean **−0.003** (12 generations) |
| Rows reduced beyond Δ < −0.01 | 5/12 |
| Visual quality | SSIM 0.963–0.984 · PSNR 40.7–41.8 dB |
| Baseline strength | **weak even for the original** (ArcFace 0.09–0.65) — FaceID v1 transfers little identity at these settings, so the attack itself is a weak test |

### What this means

The real-swap identity-transfer reduction (−0.009) is *small but real and in the
right direction*, and it survives JPEG/resize. It is NOT the "substantial
reduction" the product goal asks for, and the report says so plainly. The reason
is the ε budget: the unified perturbation is bounded to ≈4.5/255 (0.0175 on the
0–1 scale) so the photo stays imperceptible. A dedicated ceiling probe (60
refinement iterations, 24 directions, 2× identity weight) moved ArcFace cosine
only 1.000 → 0.983 at SSIM 0.962 — the pixel-space identity budget, not the
iteration count, is the binding constraint. Methods like ID-Eraser escape this
limit by *reconstructing* the face from a heavily perturbed identity vector
(their learned decoder is the missing piece here; see §10).

## 10. Known limitations

- The reconstruction/decoder half of ID-Eraser (a learned "Face Revive
  Generator" producing a natural protected face from the perturbed identity
  representation) is **not implemented** — we optimize directly in pixel space
  with a bounded δ, which caps how far identity can move before quality breaks.
  This is the single biggest reason identity transfer only moves ~1–2%.
- Phantom's full latent+spatial dual constraint is not reproduced; we use the
  spatial (elliptical region) constraint only.
- INSwapper and IP-Adapter FaceID are non-commercial/restricted for
  redistribution; they are evaluation-only and never shipped in production.
- Face-swap protection is bounded by the visual-quality constraint — if pushing
  identity further would visibly damage the photo, the optimizer stops.
