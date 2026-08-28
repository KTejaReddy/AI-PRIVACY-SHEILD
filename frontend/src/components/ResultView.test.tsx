import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ProcessResult } from "../types";
import { ResultView } from "./ResultView";

const PNG_1PX =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

function makeResult(overrides: Partial<ProcessResult> = {}): ProcessResult {
  return {
    session_id: "s1",
    width: 1,
    height: 1,
    face_count: 2,
    faces: [],
    person_count: 1,
    persons: [{ x1: 0, y1: 0, x2: 10, y2: 20, confidence: 1.2 }],
    faces_message: "2 faces detected — protecting all detected faces.",
    protection_applied: true,
    protection: { applied: true, iterations: 20, message: "protected" },
    provenance: { available: true, enabled: true, applied: true, note: "C2PA manifest embedded." },
    sensitive: { regions: [], qr_codes: [], text_regions: [], ocr_available: false, ocr_note: null, experimental: false, summary: "none" },
    metadata: { source_had_exif: true, source_had_gps: false, source_had_xmp: false, removed: ["EXIF"], output_format: "png", note: "cleaned", source: {} },
    quality: { psnr_db: 36.5, ssim: 0.967, mse: 3.8, mae: 1.2, perturbation_l2: 120, perturbation_linf: 9, lpips: null },
    robustness: {
      overall: "PARTIAL",
      transforms: {
        jpeg_compression: { verdict: "PARTIAL", per_model: { a: 0.5 }, mean: 0.51 },
        resize: { verdict: "FAIL", per_model: { a: 0.35 }, mean: 0.35 },
      },
      base_distances: { a: 0.7 },
      model_summary: { a: { mean_distance: 0.5, base_distance: 0.7, verdict: "PARTIAL" } },
      thresholds: { pass: 0.7, partial: 0.4, unit: "L2" },
      note: "note",
    },
    perception: {
      faces: { tested: true, detected: 2, before: 0.96, after: 0.17, change_pct: -82.3 },
      persons: { tested: true, detected: 1, before: 1.2, after: 0.4, change_pct: -66.7 },
      embeddings: {
        facenet_vggface2: { display_name: "FaceNet (VGGFace2)", before: 1.0, after: 0.31, mean_distance: 0.92, change_pct: -69.0, tested: true },
      },
      note: "Measured on the protected image against the tested detectors and models.",
    },
    editing: {
      enabled: true,
      protection: {
        applied: true,
        iterations: 6,
        denoising_loss_before: 1.8,
        denoising_loss_after: 2.2,
        loss_increase_pct: 22.2,
        note: "High-frequency anti-diffusion perturbation applied.",
      },
      benchmark: {
        available: true,
        tasks: [
          { id: "t1", name: "Shirt Color", editor_type: "instruction", instruction: "Make shirt red", target: "a red shirt", mask_kind: null, task_metric_original: 0.3, task_metric_protected: 0.1, clip_delta_original: 0.05, clip_delta_protected: 0.01, success_original: 0.05, success_protected: 0.02, absolute_change: 0.03, relative_change_pct: 60.0, semantic_preservation_original: 0.9, semantic_preservation_protected: 0.9, edit_magnitude_original: 0.03, edit_magnitude_protected: 0.03, protected_region_change_original: 0.01, protected_region_change_protected: 0.01 },
        ],
        aggregate: { mean_original: 0.05, mean_protected: 0.02, mean_absolute_change: 0.03, mean_relative_change_pct: 60.0, tasks_reduced: 1, tasks_total: 1, mean_reduction_pct: 60.0 },
        robustness: [
          { transform: "jpeg_compression", tasks: [], mean_success_original: 0.05, mean_success_after_transform: 0.03 },
        ],
        note: "Same editor, same instruction, only the input image changes.",
      },
    },
    vlm: { enabled: false, note: "VLM semantic protection is not enabled in this configuration." },
    processing_time_ms: 4210,
    models: { device: "cuda", models: [{ id: "facenet_vggface2", display_name: "FaceNet", kind: "optimization", architecture: "x", license_note: "", loaded: true, error: null, device: "cuda", state: "ok" }] },
    hardware: { device: "cuda", cuda: true, gpu_name: "RTX", note: "GPU acceleration: available" },
    original_data_url: PNG_1PX,
    protected_data_url: PNG_1PX,
    output_format: "png",
    messages: [],
    ...overrides,
  };
}

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true }));
  Object.defineProperty(window, "localStorage", {
    value: { getItem: vi.fn(() => null) },
    writable: true,
  });
  URL.createObjectURL = vi.fn(() => "blob:created");
  URL.revokeObjectURL = vi.fn();
  // ensure clipboard globals do not leak between tests
  delete (globalThis as Record<string, unknown>).ClipboardItem;
  delete (navigator as unknown as Record<string, unknown>).clipboard;
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("ResultView", () => {
  it("renders the three comparison panels, plain-language summary and actions", () => {
    render(<ResultView result={makeResult()} onActionDone={() => {}} onNewImage={() => {}} />);
    expect(screen.getAllByText("Original").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Protected").length).toBeGreaterThan(0);
    expect(screen.getByText("Difference")).toBeInTheDocument();
    expect(screen.getByText(/Your photo has been protected/)).toBeInTheDocument();
    expect(screen.getByText("Technical details (research / developer view)")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Copy Image" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Download Image" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Start New Image" })).toBeInTheDocument();
  });

  it("keeps technical verdicts inside the collapsed details", () => {
    render(<ResultView result={makeResult()} onActionDone={() => {}} onNewImage={() => {}} />);
    // the research/developer view is collapsed by default: the details
    // element must be closed so verdicts are not part of the visible flow
    const details = document.querySelector(".technical-details");
    expect(details).not.toBeNull();
    expect(details).not.toHaveAttribute("open");
  });

  it("copies the protected image and reports cleanup", async () => {
    const clipboardWrite = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", {
      value: { write: clipboardWrite },
      writable: true,
      configurable: true,
    });
    Object.defineProperty(globalThis, "ClipboardItem", {
      value: class ClipboardItem {
        constructor(public items: Record<string, Blob>) {}
      },
      writable: true,
      configurable: true,
    });

    const onActionDone = vi.fn();
    render(<ResultView result={makeResult()} onActionDone={onActionDone} onNewImage={() => {}} />);
    fireEvent.click(screen.getByRole("button", { name: "Copy Image" }));

    await vi.waitFor(() => expect(clipboardWrite).toHaveBeenCalledTimes(1));
    await vi.waitFor(() => expect(onActionDone).toHaveBeenCalled());
    const outcome = onActionDone.mock.calls[0][0];
    expect(outcome.ok).toBe(true);
    expect(outcome.message).toMatch(/temporary image data is cleared after copy/i);
  });

  it("falls back gracefully when the clipboard API is unsupported", async () => {
    // ClipboardItem is undefined in this environment by default
    const onActionDone = vi.fn();
    render(<ResultView result={makeResult()} onActionDone={onActionDone} onNewImage={() => {}} />);
    fireEvent.click(screen.getByRole("button", { name: "Copy Image" }));
    await vi.waitFor(() => expect(onActionDone).toHaveBeenCalled());
    const outcome = onActionDone.mock.calls[0][0];
    expect(outcome.ok).toBe(false);
    expect(outcome.message).toMatch(/Copy is not supported/);
  });

  it("downloads the protected image, revokes the object URL and reports cleanup", async () => {
    const createSpy = vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:download");
    const revokeSpy = vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});

    const onActionDone = vi.fn();
    render(<ResultView result={makeResult()} onActionDone={onActionDone} onNewImage={() => {}} />);
    fireEvent.click(screen.getByRole("button", { name: "Download Image" }));

    await vi.waitFor(() => expect(onActionDone).toHaveBeenCalled());
    expect(createSpy).toHaveBeenCalledTimes(1);
    expect(revokeSpy).toHaveBeenCalled();
    const outcome = onActionDone.mock.calls[0][0];
    expect(outcome.ok).toBe(true);
    expect(outcome.message).toMatch(/cleared after download/i);
  });

  it("invokes onNewImage from the start-new button", () => {
    const onNewImage = vi.fn();
    render(<ResultView result={makeResult()} onActionDone={() => {}} onNewImage={onNewImage} />);
    fireEvent.click(screen.getByRole("button", { name: "Start New Image" }));
    expect(onNewImage).toHaveBeenCalledTimes(1);
  });
});
