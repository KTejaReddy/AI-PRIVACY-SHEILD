"""AI-editing protection: anti-diffusion surrogate + local editing benchmark.

The project's primary objective is that a human sees and uses the protected
photo normally while a tested AI image editor has substantially reduced
ability to use it as a source for unauthorized editing. This module provides:

  * ``tasks``        — controlled, reproducible editing tasks
  * ``adapter``      — local editing model adapters (InstructPix2Pix)
  * ``surrogate``    — differentiable anti-diffusion objective (Photoguard-style)
  * ``manager``      — one-at-a-time model loading within the VRAM budget
"""
