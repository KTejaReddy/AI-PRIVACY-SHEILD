"""End-to-end processing pipeline (consolidated).

Stages (each emits a real event):

    analyze    -> load/validate/orient the upload
    faces      -> face detection
    sensitive  -> QR / OCR / PII / metadata audit
    protect    -> ONE multi-family perturbation (diffusion editing + inpainting +
                  instruction editing + img2img + identity-reference + face-swap +
                  vision encoders) via the differentiable anti-diffusion PGD
    treat      -> pixelate sensitive regions
    test       -> quality (research profile also runs perception/robustness)
    finalize   -> metadata sanitization + optional C2PA provenance + encoding
    cleanup    -> delete temp files
    done / error

Cleanup of the session directory is guaranteed in ``finally`` — on success,
on failure, and on client disconnect.

Consolidation notes (spec §12, §21-23, §25-26)
-----------------------------------------------
* There is exactly ONE protection stage. The old stacked face-protection pass
  (adversarial/protector.py) was moved to ``backend/legacy/``: it applied a
  second, independent perturbation on top of the multi-family PGD, which
  degraded visual quality (production SSIM fell to ~0.94 vs ~0.98 for the
  same algorithm alone). The unified PGD already includes the face-identity
  objective, so production now produces the same perturbation the research
  benchmark measures.
* Perception/robustness testing loads research verification models (CASIA,
  ArcFace, ResNet50). In the production profile the test stage measures
  visual quality only; the full evaluation runs in the research profile.
* The editing benchmark (3 editors) is research tooling: skipped in the
  production profile (see configs/production.yaml).
* C2PA provenance is embedded at finalize as a second, non-AI-blocking layer.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import numpy as np

from ..cleanup.manager import get_store
from ..config import settings
from ..editing.protector import apply_editing_protection
from ..evaluation.perception import PerceptionEngine
from ..metadata.provenance import add_c2pa_manifest
from ..metadata.sanitizer import sanitize_and_encode
from ..models.face_models import get_registry
from ..privacy.analyzer import PrivacyAnalyzer
from ..quality.metrics import compute_quality
from ..robustness.tester import RobustnessTester
from ..utils.imaging import ImageRecord, load_and_normalize, to_data_url

logger = logging.getLogger(__name__)

Emit = Callable[[dict], None]

# Human-readable stage names (shown in the UI progress panel).
STAGE_LABELS = {
    "analyze": "Analyzing image",
    "faces": "Detecting faces",
    "sensitive": "Analyzing sensitive regions",
    "protect": "Protecting photo against AI manipulation",
    "treat": "Treating sensitive regions",
    "test": "Testing protection",
    "editing_benchmark": "Running AI-editing benchmark",
    "finalize": "Sanitizing metadata & preparing protected image",
    "cleanup": "Clearing temporary data",
}


def _emit(emit: Emit, ev_type: str, **kwargs) -> None:
    try:
        emit({"type": ev_type, **kwargs})
    except Exception:  # noqa: BLE001 - emitter must never crash the pipeline
        pass


def run_pipeline(session_id: str, emit: Emit) -> dict:
    """Run the full pipeline for a session. Returns the result payload.

    Temp files are always deleted before returning, whether it succeeds or not.
    """
    store = get_store()
    device = settings.DEVICE
    messages: list[str] = []
    protected: np.ndarray | None = None
    original: np.ndarray | None = None
    t0 = time.perf_counter()

    try:
        # ---- models (loaded once; warm-up) -------------------------------
        registry = get_registry(device)
        if not any(m.info.loaded for m in registry.optimization_models):
            _emit(emit, "stage", stage="protect", message=STAGE_LABELS["protect"], loading_models=True)
            registry.load_all()

        # ---- analyze -----------------------------------------------------
        _emit(emit, "stage", stage="analyze", message=STAGE_LABELS["analyze"])
        raw = store.upload_path(session_id).read_bytes()
        record: ImageRecord = load_and_normalize(raw, len(raw))
        original = record.array
        h, w = record.height, record.width
        _emit(emit, "stage_done", stage="analyze", message=f"Validated {record.source_format.upper()} image ({w}×{h}).")

        # ---- faces -------------------------------------------------------
        _emit(emit, "stage", stage="faces", message=STAGE_LABELS["faces"])
        analyzer = PrivacyAnalyzer()
        analysis = analyzer.analyze(record)
        face_count = analysis.face_count
        faces = analysis.faces
        person_count = analysis.person_count
        persons = analysis.persons
        person_count_hog = analysis.person_count_hog
        person_count_neural = analysis.person_count_neural
        detector_note = (
            f"HOG: {person_count_hog}, neural (Faster R-CNN): {person_count_neural}"
            if analysis.neural_available
            else f"HOG: {person_count_hog}, neural detector unavailable"
        )
        persons_msg = (
            f" {person_count} person(s) detected ({detector_note}) — person regions included in the protection mask."
            if person_count
            else " No person detected (HOG and neural detectors)."
        )
        _emit(emit, "stage_done", stage="faces", message=f"{face_count} face(s) detected.{persons_msg}")
        _emit(
            emit,
            "faces",
            count=face_count,
            faces=[f.as_dict() for f in faces],
            person_count=person_count,
            persons=[p.as_dict() for p in persons],
            person_count_hog=person_count_hog,
            person_count_neural=person_count_neural,
            neural_available=analysis.neural_available,
            message=(
                f"{face_count} face(s) detected — protecting all detected faces.{persons_msg}"
                if face_count
                else f"No face detected. No facial-identity protection was applied.{persons_msg}"
            ),
        )

        # ---- sensitive regions ------------------------------------------
        _emit(emit, "stage", stage="sensitive", message=STAGE_LABELS["sensitive"])
        sensitive = analysis.sensitive
        _emit(emit, "stage_done", stage="sensitive", message=sensitive["summary"])

        # ---- protect (ONE unified multi-family perturbation) -------------
        # The only protection stage: a single PGD perturbation targeting
        # diffusion editing / inpainting / instruction editing / img2img
        # (anti-diffusion surrogate), identity-reference / face-swap (FaceNet
        # on the detected face boxes) and vision encoders (MobileNetV3)
        # simultaneously. No second pass is stacked on top.
        protection_applied = False
        protect_result = None
        try:
            from ..editing.manager import get_editing_manager  # noqa: PLC0415

            if settings.EDITING_ENABLED and settings.EDITING_SURROGATE_ENABLED:
                editing_manager = get_editing_manager(registry.device)
                if editing_manager.surrogate_available():
                    _emit(emit, "stage", stage="protect", message=STAGE_LABELS["protect"], total=1)
                    editing_manager.offload_registry(registry)
                    try:

                        def on_protect_progress(info: dict) -> None:
                            _emit(emit, "progress", **info)

                        protect_result = apply_editing_protection(
                            original,
                            registry.device,
                            progress=on_protect_progress,
                            face_boxes=[f.box for f in faces],
                        )
                        if protect_result.applied and protect_result.protected is not None:
                            protected = protect_result.protected
                            protection_applied = True
                        _emit(emit, "stage_done", stage="protect", message=protect_result.note)
                    finally:
                        editing_manager.unload_all()
                        editing_manager.restore_registry(registry)
                else:
                    _emit(
                        emit,
                        "stage_done",
                        stage="protect",
                        message=(
                            "AI-manipulation protection not run: the local anti-diffusion "
                            "surrogate (SD1.5) is not available on this hardware. "
                            "See docs/editing-protection.md."
                        ),
                    )
            else:
                _emit(
                    emit,
                    "stage_done",
                    stage="protect",
                    message="AI-manipulation protection disabled by configuration.",
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unified protection stage failed")
            _emit(emit, "stage_done", stage="protect", message=f"AI-manipulation protection failed: {_user_safe_error(exc)}")

        # ---- treat sensitive regions -------------------------------------
        _emit(emit, "stage", stage="treat", message=STAGE_LABELS["treat"])
        if protected is None:
            protected = original.copy()
        treated = analyzer.treat_sensitive_regions(protected, sensitive["regions"])
        _emit(emit, "stage_done", stage="treat", message=f"{treated} sensitive region(s) pixelated.")

        # ---- test --------------------------------------------------------
        _emit(emit, "stage", stage="test", message=STAGE_LABELS["test"])
        quality = compute_quality(original, protected, device=registry.device)
        robustness = None
        perception = None
        if settings.PROFILE == "research":
            # Perception/robustness load research verification models; they
            # run only in the research profile (see configs/research.yaml).
            tester = RobustnessTester(registry, registry.device)
            robustness = tester.evaluate(original, protected, faces)
            perception = PerceptionEngine(registry, registry.device).evaluate(
                original, protected, faces, persons, analysis.persons_neural
            )
        if robustness is not None:
            _emit(
                emit,
                "stage_done",
                stage="test",
                message=(
                    f"Protection tested. SSIM {quality.ssim:.3f}, PSNR {quality.psnr_db:.1f} dB, "
                    f"robustness overall: {robustness.overall}. "
                    f"Face confidence {perception.faces.get('before', '—')} → "
                    f"{perception.faces.get('after', '—')}."
                ),
            )
        else:
            _emit(
                emit,
                "stage_done",
                stage="test",
                message=(
                    f"Visual similarity measured (SSIM {quality.ssim:.3f}, PSNR {quality.psnr_db:.1f} dB)."
                    + (" Full perception/robustness evaluation is research tooling." if settings.PROFILE != "research" else "")
                ),
            )

        # ---- AI-editing benchmark stage (research profile only) -----------
        editing_block: dict = {
            "enabled": settings.EDITING_ENABLED,
            "protection": _protection_summary(protect_result),
            "benchmark": {
                "available": False,
                "note": (
                    "AI-editing benchmark is research tooling and is not run in the production "
                    "profile. Run `python scripts/benchmark_protection.py` for the full "
                    "multi-family evaluation."
                ),
            },
            "robustness": [],
        }
        if settings.PROFILE == "research" and settings.EDITING_ENABLED and settings.EDITING_BENCHMARK_ENABLED:
            try:
                from ..editing.manager import get_editing_manager  # noqa: PLC0415
                from ..evaluation.editing_benchmark import EditingBenchmark  # noqa: PLC0415

                editing_manager = get_editing_manager(registry.device)
                if editing_manager.editing_available():
                    _emit(
                        emit,
                        "stage",
                        stage="editing_benchmark",
                        message=STAGE_LABELS["editing_benchmark"],
                    )
                    editing_manager.offload_registry(registry)
                    try:

                        def on_bench_progress(info: dict) -> None:
                            _emit(emit, "progress", **info)

                        bench = EditingBenchmark(registry.device)
                        bench_result = bench.run(
                            original,
                            protected,
                            face_boxes=[f.box for f in faces],
                            person_boxes=[p.box for p in persons],
                            progress=on_bench_progress,
                        )
                        editing_block["benchmark"] = bench_result.as_dict()
                        # ---- edit success under real transformations ----
                        if settings.EDITING_ROBUSTNESS_ENABLED and bench_result.available:
                            editing_block["robustness"] = bench.run_robustness(
                                original,
                                protected,
                                face_boxes=[f.box for f in faces],
                                person_boxes=[p.box for p in persons],
                                progress=on_bench_progress,
                            )
                        else:
                            editing_block["robustness"] = []
                        if bench_result.available and bench_result.tasks:
                            rel = (
                                f" ({bench_result.mean_relative_change_pct:.0f}% relative)"
                                if bench_result.mean_relative_change_pct is not None
                                else ""
                            )
                            msg = (
                                f"AI-editing benchmark: {len(bench_result.tasks)} task/editor rows, "
                                f"mean edit success {bench_result.mean_original:.3f} → "
                                f"{bench_result.mean_protected:.3f} "
                                f"(absolute change {bench_result.mean_absolute_change:.3f}{rel}). "
                                f"{bench_result.tasks_reduced}/{bench_result.tasks_total} rows reduced."
                            )
                            rob = editing_block["robustness"]
                            if rob:
                                parts = []
                                for r in rob:
                                    if "error" in r:
                                        parts.append(f"{r['transform']}: error")
                                    else:
                                        parts.append(
                                            f"{r['transform']}: success {r['mean_success_original']:.3f} → "
                                            f"{r['mean_success_after_transform']:.3f}"
                                        )
                                msg += " Robustness — " + "; ".join(parts) + "."
                        else:
                            msg = bench_result.note
                        _emit(emit, "stage_done", stage="editing_benchmark", message=msg)
                    finally:
                        editing_manager.unload_all()
                        editing_manager.restore_registry(registry)
                else:
                    editing_block["benchmark"] = {
                        "available": False,
                        "note": (
                            "AI-editing benchmark unavailable on this hardware: the local editing "
                            "model (InstructPix2Pix) or the CLIP scorer are not downloaded. "
                            "See docs/editing-protection.md for installation."
                        ),
                    }
            except Exception as exc:  # noqa: BLE001
                logger.warning("Editing benchmark failed: %s", exc)
                editing_block["benchmark"] = {
                    "available": False,
                    "note": f"AI-editing benchmark failed: {_user_safe_error(exc)}",
                }
                _emit(
                    emit,
                    "stage_done",
                    stage="editing_benchmark",
                    message=editing_block["benchmark"]["note"],
                )

        # ---- finalize ----------------------------------------------------
        _emit(emit, "stage", stage="finalize", message=STAGE_LABELS["finalize"])
        protected_bytes, metadata_report = sanitize_and_encode(
            protected,
            source_had_exif=record.had_exif,
            source_had_gps=record.had_gps,
            source_had_xmp=record.had_xmp,
        )
        provenance = add_c2pa_manifest(
            protected_bytes, width=w, height=h, output_format=settings.OUTPUT_FORMAT
        )
        if provenance.applied:
            protected_bytes = provenance.signed_bytes
        original_bytes = _encode_original(original)
        _emit(emit, "stage_done", stage="finalize", message=metadata_report.note)

        # ---- cleanup + result --------------------------------------------
        _emit(emit, "stage", stage="cleanup", message=STAGE_LABELS["cleanup"])
        result = {
            "session_id": session_id,
            "width": w,
            "height": h,
            "face_count": face_count,
            "faces": [f.as_dict() for f in faces],
            "person_count": person_count,
            "persons": [p.as_dict() for p in persons],
            "person_count_hog": person_count_hog,
            "person_count_neural": person_count_neural,
            "neural_available": analysis.neural_available,
            "faces_message": (
                f"{face_count} face(s) detected — protecting all detected faces.{persons_msg}"
                if face_count
                else f"No face detected. No facial-identity protection was applied.{persons_msg}"
            ),
            "protection_applied": protection_applied,
            "protection": editing_block["protection"],
            "sensitive": sensitive,
            "metadata": metadata_report.as_dict() | {"source": analysis.metadata},
            "quality": quality.as_dict(),
            "robustness": robustness.as_dict() if robustness is not None else None,
            "perception": perception.as_dict() if perception is not None else None,
            "editing": editing_block,
            "families": _families_report(),
            "profile": settings.PROFILE,
            "provenance": {
                "available": provenance.available,
                "enabled": provenance.enabled,
                "applied": provenance.applied,
                "note": provenance.note,
            },
            "vlm": {
                "enabled": False,
                "note": "VLM semantic protection is not enabled in this configuration.",
            },
            "processing_time_ms": int(round((time.perf_counter() - t0) * 1000)),
            "reproducibility": {
                "seed": settings.OPT_SEED,
                "epsilon": settings.PERTURBATION_EPSILON,
                "opt_iterations_gpu": settings.OPT_ITERATIONS_GPU,
                "opt_iterations_cpu": settings.OPT_ITERATIONS_CPU,
                "editing_surrogate_iters_gpu": settings.EDITING_SURROGATE_ITERS_GPU,
                "editing_surrogate_iters_cpu": settings.EDITING_SURROGATE_ITERS_CPU,
                "thresholds": {
                    "min_ssim": settings.MIN_SSIM,
                    "min_psnr": settings.MIN_PSNR,
                    "editing_min_ssim": settings.EDITING_MIN_SSIM,
                },
            },
            "models": registry.describe(),
            "hardware": {
                "device": registry.device,
                "cuda": registry.device == "cuda",
                "gpu_name": _gpu_name(registry.device),
                "note": "GPU acceleration: available" if registry.device == "cuda" else "CPU mode: active",
            },
            "original_data_url": to_data_url(original_bytes, "image/png"),
            "protected_data_url": to_data_url(protected_bytes, "image/png" if settings.OUTPUT_FORMAT == "png" else "image/jpeg"),
            "output_format": settings.OUTPUT_FORMAT,
            "messages": messages,
        }
        _emit(emit, "result", **result)
        _emit(emit, "done", ok=True, message="Processing complete.")
        return result
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline failed for session %s", session_id)
        _emit(
            emit,
            "error",
            message=_user_safe_error(exc),
        )
        raise
    finally:
        # guaranteed temp cleanup
        try:
            store.delete(session_id)
        except Exception:  # noqa: BLE001
            logger.warning("Cleanup failed for session %s", session_id, exc_info=True)


def _encode_original(arr: np.ndarray) -> bytes:
    from ..utils.imaging import array_to_png_bytes

    return array_to_png_bytes(arr)


def _protection_summary(protect_result) -> dict:
    """Summarize the unified multi-family protection result for the payload."""
    if protect_result is None:
        return {
            "applied": False,
            "message": "No AI-manipulation protection was applied (protection stage unavailable).",
        }
    if not protect_result.applied:
        return {"applied": False, "message": protect_result.note}
    return {
        "applied": True,
        "iterations": protect_result.iterations,
        "epsilon": protect_result.epsilon,
        "resolution": protect_result.resolution,
        "timestep": protect_result.timestep,
        "families": protect_result.families,
        "denoising_loss_before": protect_result.denoising_loss_before,
        "denoising_loss_after": protect_result.denoising_loss_after,
        "identity_similarity_before": protect_result.identity_similarity_before,
        "identity_similarity_after": protect_result.identity_similarity_after,
        "vision_similarity_before": protect_result.vision_similarity_before,
        "vision_similarity_after": protect_result.vision_similarity_after,
        "reverted": protect_result.reverted,
        "message": protect_result.note,
    }


def _families_report() -> dict:
    """Attack-family summary for the current profile (spec §29)."""
    try:
        from ..attack_registry import families_report  # noqa: PLC0415

        return families_report(settings.PROFILE)
    except Exception:  # noqa: BLE001
        logger.warning("Attack-family report unavailable", exc_info=True)
        return {"profile": settings.PROFILE, "families": [], "flags": {}}


def _gpu_name(device: str) -> str | None:
    if device != "cuda":
        return None
    try:
        import torch  # noqa: PLC0415

        return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        return None


def _user_safe_error(exc: Exception) -> str:
    """Map exceptions to user-safe messages (never raw stack traces)."""
    text = str(exc)
    if "Protection model unavailable" in text:
        return text
    if "Face detection model files are missing" in text:
        return text
    if "could not be decoded" in text or "Unsupported or unrecognized" in text:
        return text
    if "exceeds the" in text:
        return text
    if "too large" in text:
        return text
    if settings.DEBUG:
        return f"Processing failed: {type(exc).__name__}: {text}"
    return "Protection failed. Your original image has not been stored."


_processing_lock = threading.Lock()


def run_pipeline_locked(session_id: str, emit: Emit) -> dict:
    """Serialize heavy processing (single local GPU/CPU) across sessions."""
    with _processing_lock:
        return run_pipeline(session_id, emit)
