"""Controlled editing tasks for the AI-editing benchmark.

Each task is *measurable*: ``instruction`` is the canonical prompt fed to the
editing model, ``target`` is the semantic description used only as an
*auxiliary* CLIP check, ``metric`` selects the region-aware pixel metric that
provides the primary edit-success evidence (see evaluation/task_metrics.py),
and ``mask_kind`` selects the region used by the masked-inpainting variant.

``prompt_variants`` are equivalent re-phrasings used to test whether the
protection overfits to one exact prompt (benchmark script only; the in-app
benchmark uses the canonical instruction to stay fast).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EditingTask:
    id: str
    name: str
    instruction: str  # prompt given to the editing model (identical for original & protected)
    target: str  # text for the auxiliary CLIP semantic check
    metric: str = "shirt_color"  # key into evaluation.task_metrics.TASK_METRICS
    mask_kind: str | None = None  # mask used by the masked-inpainting attack
    prompt_variants: tuple[str, ...] = ()


# Task ids are stable across runs so benchmark reports can be compared.
DEFAULT_TASKS: tuple[EditingTask, ...] = (
    EditingTask(
        id="t01_shirt_color",
        name="Change shirt color",
        instruction="Make the shirt red.",
        target="a person wearing a red shirt",
        metric="shirt_color",
        mask_kind="shirt",
        prompt_variants=(
            "Make the shirt red.",
            "Change the shirt color to red.",
            "Replace the shirt color with red.",
        ),
    ),
    EditingTask(
        id="t02_background",
        name="Change background",
        instruction="Make the background a beach.",
        target="a person at a sunny beach",
        metric="background",
        mask_kind="background",
        prompt_variants=(
            "Make the background a beach.",
            "Change the background to a sunny beach.",
            "Put a beach behind the person.",
        ),
    ),
    EditingTask(
        id="t03_hat",
        name="Add a hat",
        instruction="Add a hat.",
        target="a person wearing a hat",
        metric="hat",
        mask_kind="hair",
        prompt_variants=(
            "Add a hat.",
            "Put a hat on the person.",
            "Give the person a hat.",
        ),
    ),
    EditingTask(
        id="t04_lighting",
        name="Change lighting",
        instruction="Make the lighting warm sunset light.",
        target="a scene lit by warm golden sunset light",
        metric="lighting",
        mask_kind=None,
        prompt_variants=(
            "Make the lighting warm sunset light.",
            "Change the lighting to warm golden hour.",
            "Light the scene with warm sunset light.",
        ),
    ),
    EditingTask(
        id="t05_sketch",
        name="Convert to pencil sketch",
        instruction="Make it a pencil sketch.",
        target="a pencil sketch drawing",
        metric="sketch",
        mask_kind="person",
        prompt_variants=(
            "Make it a pencil sketch.",
            "Convert the photo to a pencil sketch drawing.",
            "Turn this into a pencil drawing.",
        ),
    ),
    EditingTask(
        id="t06_hairstyle",
        name="Change hairstyle",
        instruction="Change the hairstyle.",
        target="a person with a different hairstyle",
        metric="hairstyle",
        mask_kind="hair",
        prompt_variants=(
            "Change the hairstyle.",
            "Give the person a different hairstyle.",
            "Restyle the hair.",
        ),
    ),
)

TASKS_BY_ID: dict[str, EditingTask] = {t.id: t for t in DEFAULT_TASKS}


def tasks_for(ids: tuple[str, ...] | list[str] | None = None) -> list[EditingTask]:
    """Resolve the configured task subset, preserving the canonical order."""
    if not ids:
        return list(DEFAULT_TASKS)
    seen: set[str] = set()
    out: list[EditingTask] = []
    for tid in ids:
        task = TASKS_BY_ID.get(tid)
        if task is not None and tid not in seen:
            out.append(task)
            seen.add(tid)
    return out
