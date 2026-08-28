# AI PRIVACY SHIELD — REFERENCE-CONDITIONED GENERATION BENCHMARK (IP-Adapter FaceID SD1.5)

Reference photo -> identity encoder -> NEW synthetic image of the same person. Identity transfer = ArcFace (buffalo_l w600k_r50) cosine of the generated face vs the ORIGINAL reference embedding. FaceNet VGGFace2 is a held-out encoder family.

## bohr
- Quality: SSIM 0.9634 · PSNR 40.7 dB · protect 0.0s
- Reference embedding (ArcFace): 1.000 -> 0.984

| Prompt | Orig ArcFace | Prot ArcFace | Δ | Orig FaceNet | Prot FaceNet |
|---|---|---:|---:|---:|---:|---:|
| professional studio portrait photo of a  | 0.1748 | 0.1634 | -0.011 | 0.043 | -0.0116 |
| outdoor casual portrait photo of a perso | 0.164 | 0.217 | +0.053 | -0.0308 | 0.0179 |
| portrait photo of a person, dramatic lig | 0.2232 | 0.2521 | +0.029 | -0.1405 | -0.1539 |

## curie
- Quality: SSIM 0.9692 · PSNR 41.56 dB · protect 0.0s
- Reference embedding (ArcFace): 1.000 -> 0.987

| Prompt | Orig ArcFace | Prot ArcFace | Δ | Orig FaceNet | Prot FaceNet |
|---|---|---:|---:|---:|---:|---:|
| professional studio portrait photo of a  | 0.3463 | 0.3453 | -0.001 | 0.5221 | 0.5555 |
| outdoor casual portrait photo of a perso | None | 0.3514 | — | None | 0.4955 |
| portrait photo of a person, dramatic lig | 0.1878 | 0.1982 | +0.010 | -0.0334 | 0.3148 |

## lincoln
- Quality: SSIM 0.9841 · PSNR 41.48 dB · protect 0.0s
- Reference embedding (ArcFace): 1.000 -> 0.986

| Prompt | Orig ArcFace | Prot ArcFace | Δ | Orig FaceNet | Prot FaceNet |
|---|---|---:|---:|---:|---:|---:|
| professional studio portrait photo of a  | 0.654 | 0.5601 | -0.094 | 0.6588 | 0.6926 |
| outdoor casual portrait photo of a perso | 0.3203 | 0.2667 | -0.054 | 0.3939 | 0.3345 |
| portrait photo of a person, dramatic lig | 0.1152 | 0.166 | +0.051 | 0.164 | 0.1779 |

## tesla
- Quality: SSIM 0.9764 · PSNR 41.81 dB · protect 0.0s
- Reference embedding (ArcFace): 1.000 -> 0.969

| Prompt | Orig ArcFace | Prot ArcFace | Δ | Orig FaceNet | Prot FaceNet |
|---|---|---:|---:|---:|---:|---:|
| professional studio portrait photo of a  | 0.2258 | 0.2751 | +0.049 | 0.0458 | 0.378 |
| outdoor casual portrait photo of a perso | 0.0886 | 0.0988 | +0.010 | -0.015 | 0.3524 |
| portrait photo of a person, dramatic lig | 0.3029 | 0.2547 | -0.048 | 0.4639 | 0.3562 |

## einstein
- Quality: SSIM 0.9798 · PSNR 41.14 dB · protect 0.0s
- Reference embedding (ArcFace): 1.000 -> 0.993

| Prompt | Orig ArcFace | Prot ArcFace | Δ | Orig FaceNet | Prot FaceNet |
|---|---|---:|---:|---:|---:|---:|
| professional studio portrait photo of a  | 0.4927 | 0.4646 | -0.028 | 0.3873 | 0.4553 |
| outdoor casual portrait photo of a perso | None | None | — | None | None |
| portrait photo of a person, dramatic lig | None | 0.3549 | — | None | 0.5116 |

## Aggregate (identity transfer, original vs protected reference)
- Generations: 12 · mean Δ: -0.003 · median Δ: +0.005 · rows reduced (Δ < -0.01): 5/12
| Source | Prompt | Orig | Prot | Δ |
|---|---|---|---:|---:|
| bohr | professional studio portrait photo of a  | 0.1748 | 0.1634 | -0.011 |
| bohr | outdoor casual portrait photo of a perso | 0.164 | 0.217 | +0.053 |
| bohr | portrait photo of a person, dramatic lig | 0.2232 | 0.2521 | +0.029 |
| curie | professional studio portrait photo of a  | 0.3463 | 0.3453 | -0.001 |
| curie | portrait photo of a person, dramatic lig | 0.1878 | 0.1982 | +0.010 |
| lincoln | professional studio portrait photo of a  | 0.654 | 0.5601 | -0.094 |
| lincoln | outdoor casual portrait photo of a perso | 0.3203 | 0.2667 | -0.054 |
| lincoln | portrait photo of a person, dramatic lig | 0.1152 | 0.166 | +0.051 |
| tesla | professional studio portrait photo of a  | 0.2258 | 0.2751 | +0.049 |
| tesla | outdoor casual portrait photo of a perso | 0.0886 | 0.0988 | +0.010 |
| tesla | portrait photo of a person, dramatic lig | 0.3029 | 0.2547 | -0.048 |
| einstein | professional studio portrait photo of a  | 0.4927 | 0.4646 | -0.028 |
