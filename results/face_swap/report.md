# AI PRIVACY SHIELD — REAL FACE-SWAP BENCHMARK (INSwapper)
Identity transfer = ArcFace (buffalo_l w600k_r50) cosine of the swapped output vs the SOURCE face — the metric swap pipelines themselves maximize. Lower on the protected source = the attack transfers less identity. FaceNet VGGFace2 is a held-out encoder family (never used by the swap).
## bohr.jpg
- Quality: SSIM 0.9634 · PSNR 40.7 dB · protect 250.7s
- Face encoders sim: 0.997 -> 0.997 ·   ArcFace w600k sim: 1.000 -> 0.986

| Target | Variant | ArcFace src sim | ArcFace tgt sim | FaceNet src sim |
|---|---|---|---|---|
| curie | original | 0.880 | 0.160 | 0.893 |
| curie | protected | 0.866 | 0.176 | 0.895 |
| curie | protected+jpeg_compression | 0.869 | 0.172 | 0.894 |
| curie | protected+resize | 0.856 | 0.166 | 0.887 |
| lincoln | original | 0.880 | 0.236 | 0.850 |
| lincoln | protected | 0.861 | 0.197 | 0.852 |
| lincoln | protected+jpeg_compression | 0.870 | 0.219 | 0.856 |
| lincoln | protected+resize | 0.856 | 0.225 | 0.840 |
| tesla | original | 0.875 | -0.026 | 0.811 |
| tesla | protected | 0.853 | -0.042 | 0.800 |
| tesla | protected+jpeg_compression | 0.867 | -0.024 | 0.800 |
| tesla | protected+resize | 0.842 | -0.040 | 0.794 |
| einstein | original | 0.860 | 0.078 | 0.868 |
| einstein | protected | 0.853 | 0.080 | 0.861 |
| einstein | protected+jpeg_compression | 0.856 | 0.077 | 0.861 |
| einstein | protected+resize | 0.827 | 0.086 | 0.857 |
## curie.jpg
- Quality: SSIM 0.9692 · PSNR 41.56 dB · protect 187.9s
- Face encoders sim: 0.989 -> 0.978 ·   ArcFace w600k sim: 1.000 -> 0.996

| Target | Variant | ArcFace src sim | ArcFace tgt sim | FaceNet src sim |
|---|---|---|---|---|
| bohr | original | 0.876 | 0.154 | 0.831 |
| bohr | protected | 0.873 | 0.150 | 0.843 |
| bohr | protected+jpeg_compression | 0.876 | 0.141 | 0.848 |
| bohr | protected+resize | 0.875 | 0.142 | 0.831 |
| lincoln | original | 0.838 | 0.186 | 0.825 |
| lincoln | protected | 0.840 | 0.196 | 0.831 |
| lincoln | protected+jpeg_compression | 0.835 | 0.197 | 0.835 |
| lincoln | protected+resize | 0.840 | 0.200 | 0.816 |
| tesla | original | 0.837 | 0.084 | 0.799 |
| tesla | protected | 0.839 | 0.087 | 0.799 |
| tesla | protected+jpeg_compression | 0.847 | 0.088 | 0.799 |
| tesla | protected+resize | 0.844 | 0.089 | 0.800 |
| einstein | original | 0.873 | 0.142 | 0.821 |
| einstein | protected | 0.877 | 0.123 | 0.819 |
| einstein | protected+jpeg_compression | 0.875 | 0.126 | 0.819 |
| einstein | protected+resize | 0.876 | 0.139 | 0.818 |
## lincoln.jpg
- Quality: SSIM 0.9841 · PSNR 41.48 dB · protect 285.1s
- Face encoders sim: 0.990 -> 0.988 ·   ArcFace w600k sim: 1.000 -> 0.998

| Target | Variant | ArcFace src sim | ArcFace tgt sim | FaceNet src sim |
|---|---|---|---|---|
| bohr | original | 0.874 | 0.265 | 0.891 |
| bohr | protected | 0.869 | 0.254 | 0.893 |
| bohr | protected+jpeg_compression | 0.854 | 0.245 | 0.881 |
| bohr | protected+resize | 0.862 | 0.255 | 0.884 |
| curie | original | 0.886 | 0.180 | 0.848 |
| curie | protected | 0.875 | 0.171 | 0.859 |
| curie | protected+jpeg_compression | 0.859 | 0.171 | 0.862 |
| curie | protected+resize | 0.863 | 0.154 | 0.861 |
| tesla | original | 0.899 | 0.071 | 0.739 |
| tesla | protected | 0.887 | 0.053 | 0.740 |
| tesla | protected+jpeg_compression | 0.860 | 0.067 | 0.763 |
| tesla | protected+resize | 0.864 | 0.055 | 0.751 |
| einstein | original | 0.887 | 0.155 | 0.817 |
| einstein | protected | 0.871 | 0.161 | 0.803 |
| einstein | protected+jpeg_compression | 0.861 | 0.146 | 0.817 |
| einstein | protected+resize | 0.858 | 0.148 | 0.800 |
## tesla.jpg
- Quality: SSIM 0.9764 · PSNR 41.81 dB · protect 166.8s
- Face encoders sim: 0.985 -> 0.979 ·   ArcFace w600k sim: 1.000 -> 0.996

| Target | Variant | ArcFace src sim | ArcFace tgt sim | FaceNet src sim |
|---|---|---|---|---|
| bohr | original | 0.859 | 0.031 | 0.830 |
| bohr | protected | 0.836 | -0.009 | 0.820 |
| bohr | protected+jpeg_compression | 0.833 | 0.041 | 0.824 |
| bohr | protected+resize | 0.846 | 0.019 | 0.816 |
| curie | original | 0.820 | 0.127 | 0.848 |
| curie | protected | 0.805 | 0.135 | 0.842 |
| curie | protected+jpeg_compression | 0.796 | 0.113 | 0.836 |
| curie | protected+resize | 0.817 | 0.098 | 0.820 |
| lincoln | original | 0.834 | 0.026 | 0.804 |
| lincoln | protected | 0.810 | 0.035 | 0.788 |
| lincoln | protected+jpeg_compression | 0.802 | 0.024 | 0.779 |
| lincoln | protected+resize | 0.812 | 0.021 | 0.787 |
| einstein | original | 0.837 | 0.047 | 0.852 |
| einstein | protected | 0.827 | 0.072 | 0.823 |
| einstein | protected+jpeg_compression | 0.811 | 0.063 | 0.827 |
| einstein | protected+resize | 0.817 | 0.059 | 0.831 |
## einstein.jpg
- Quality: SSIM 0.9798 · PSNR 41.14 dB · protect 126.4s
- Face encoders sim: 0.993 -> 0.992 ·   ArcFace w600k sim: 1.000 -> 0.996

| Target | Variant | ArcFace src sim | ArcFace tgt sim | FaceNet src sim |
|---|---|---|---|---|
| bohr | original | 0.862 | 0.065 | 0.813 |
| bohr | protected | 0.869 | 0.048 | 0.818 |
| bohr | protected+jpeg_compression | 0.857 | 0.062 | 0.808 |
| bohr | protected+resize | 0.857 | 0.066 | 0.804 |
| curie | original | 0.868 | 0.100 | 0.713 |
| curie | protected | 0.863 | 0.083 | 0.726 |
| curie | protected+jpeg_compression | 0.859 | 0.089 | 0.714 |
| curie | protected+resize | 0.866 | 0.084 | 0.705 |
| lincoln | original | 0.849 | 0.156 | 0.674 |
| lincoln | protected | 0.847 | 0.173 | 0.695 |
| lincoln | protected+jpeg_compression | 0.840 | 0.171 | 0.695 |
| lincoln | protected+resize | 0.840 | 0.170 | 0.685 |
| tesla | original | 0.859 | 0.035 | 0.691 |
| tesla | protected | 0.859 | 0.029 | 0.701 |
| tesla | protected+jpeg_compression | 0.851 | 0.051 | 0.693 |
| tesla | protected+resize | 0.856 | 0.048 | 0.689 |

## Aggregate (ArcFace identity transfer, original vs protected)
- Pairs: 20 · mean change: -0.009 · median change: -0.009 · rows reduced (Δ < -0.01): 9/20
| Source | Target | Orig | Prot | Δ |
|---|---|---|---:|---:|
| bohr.jpg | curie | 0.880 | 0.866 | -0.014 |
| bohr.jpg | einstein | 0.860 | 0.853 | -0.008 |
| bohr.jpg | lincoln | 0.880 | 0.861 | -0.018 |
| bohr.jpg | tesla | 0.875 | 0.853 | -0.022 |
| curie.jpg | bohr | 0.876 | 0.873 | -0.003 |
| curie.jpg | einstein | 0.873 | 0.877 | +0.004 |
| curie.jpg | lincoln | 0.838 | 0.840 | +0.002 |
| curie.jpg | tesla | 0.837 | 0.839 | +0.002 |
| einstein.jpg | bohr | 0.862 | 0.869 | +0.007 |
| einstein.jpg | curie | 0.868 | 0.863 | -0.005 |
| einstein.jpg | lincoln | 0.849 | 0.847 | -0.003 |
| einstein.jpg | tesla | 0.859 | 0.859 | -0.001 |
| lincoln.jpg | bohr | 0.874 | 0.869 | -0.005 |
| lincoln.jpg | curie | 0.886 | 0.875 | -0.012 |
| lincoln.jpg | einstein | 0.887 | 0.871 | -0.016 |
| lincoln.jpg | tesla | 0.899 | 0.887 | -0.012 |
| tesla.jpg | bohr | 0.859 | 0.836 | -0.024 |
| tesla.jpg | curie | 0.820 | 0.805 | -0.015 |
| tesla.jpg | einstein | 0.837 | 0.827 | -0.009 |
| tesla.jpg | lincoln | 0.834 | 0.810 | -0.024 |
