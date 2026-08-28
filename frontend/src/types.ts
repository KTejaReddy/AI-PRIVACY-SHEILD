// Types mirroring the backend's JSON result payload (app/processing/pipeline.py).

export interface UploadResponse {
  session_id: string;
  width: number;
  height: number;
  format: string;
  size_bytes: number;
}

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  hardware: {
    device: string;
    cuda: boolean;
    gpu_name: string | null;
    note: string;
  };
  models: {
    device: string;
    models: Array<{
      id: string;
      display_name: string;
      kind: string;
      architecture: string;
      license_note: string;
      loaded: boolean;
      error: string | null;
      device: string | null;
      state: string;
    }>;
  };
  ocr: { available: boolean; note: string | null };
}

export interface FaceBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  confidence: number;
}

export interface ProtectionInfo {
  applied: boolean;
  iterations?: number;
  epsilon?: number;
  resolution?: number;
  timestep?: number;
  families?: string[];
  denoising_loss_before?: number;
  denoising_loss_after?: number;
  identity_similarity_before?: number | null;
  identity_similarity_after?: number | null;
  vision_similarity_before?: number | null;
  vision_similarity_after?: number | null;
  reverted?: boolean;
  message: string;
}

export interface ProvenanceInfo {
  available: boolean;
  enabled: boolean;
  applied: boolean;
  note: string;
}

export interface RobustnessTransformResult {
  verdict: "PASS" | "PARTIAL" | "FAIL";
  per_model: Record<string, number>;
  mean: number;
  error?: string;
}

export interface PerceptionFaces {
  tested: boolean;
  detected: number;
  before: number | null;
  after: number | null;
  per_face_before?: number[];
  per_face_after?: number[];
  change_pct: number | null;
}

export interface PerceptionPersons {
  tested: boolean;
  detected: number;
  before: number | null;
  after: number | null;
  change_pct: number | null;
}

export interface PerceptionEmbedding {
  display_name: string;
  before: number;
  after: number;
  mean_distance: number;
  change_pct: number | null;
  tested: boolean;
}

export interface Perception {
  faces: PerceptionFaces;
  faces_mtcnn?: PerceptionFaces;
  persons: PerceptionPersons;
  persons_neural?: PerceptionPersons;
  embeddings: Record<string, PerceptionEmbedding>;
  note: string;
}

export interface EditingProtectionInfo {
  applied: boolean;
  iterations?: number;
  epsilon?: number;
  resolution?: number;
  timestep?: number;
  denoising_loss_before?: number;
  denoising_loss_after?: number;
  loss_increase_pct?: number;
  verified_loss_after?: number;
  verified_increase_pct?: number;
  reverted?: boolean;
  surrogate_model?: string;
  objective?: string;
  note: string;
  families?: string[];
  weights?: Record<string, number>;
  identity_similarity_before?: number | null;
  identity_similarity_after?: number | null;
  vision_similarity_before?: number | null;
  vision_similarity_after?: number | null;
}

export interface EditingTaskResult {
  id: string;
  name: string;
  editor_type: string;
  instruction: string;
  target: string;
  mask_kind: string | null;
  task_metric_original: number;
  task_metric_protected: number;
  clip_delta_original: number;
  clip_delta_protected: number;
  success_original: number;
  success_protected: number;
  absolute_change: number;
  relative_change_pct: number | null;
  semantic_preservation_original: number;
  semantic_preservation_protected: number;
  edit_magnitude_original: number;
  edit_magnitude_protected: number;
  protected_region_change_original: number;
  protected_region_change_protected: number;
  samples?: number;
  stats?: {
    success_original: Record<string, number>;
    success_protected: Record<string, number>;
    absolute_change: Record<string, number>;
  };
}

export interface EditingRobustnessRow {
  transform: string;
  tasks?: Array<{
    task_id: string;
    name: string;
    success_original: number;
    success_after_transform: number;
  }>;
  mean_success_original?: number;
  mean_success_after_transform?: number;
  error?: string;
}

export interface EditingBenchmark {
  available: boolean;
  model?: string;
  score_model?: string;
  settings?: {
    resolution: number;
    steps: number;
    guidance_scale: number;
    image_guidance_scale: number;
    seed: number;
    task_metric_weight: number;
    clip_weight: number;
    clip_delta_scale: number;
  };
  tasks?: EditingTaskResult[];
  robustness?: EditingRobustnessRow[];
  aggregate?: {
    mean_original: number;
    mean_protected: number;
    mean_absolute_change: number;
    mean_relative_change_pct: number | null;
    tasks_reduced: number;
    tasks_total: number;
    mean_reduction_pct: number | null;
  };
  note: string;
}

export interface EditingInfo {
  enabled: boolean;
  protection: EditingProtectionInfo;
  benchmark: EditingBenchmark;
  robustness?: EditingRobustnessRow[];
}

export interface AttackFamilyModel {
  id: string;
  name: string;
  role: string;
  local: boolean;
  note: string;
}

export interface AttackFamily {
  id: string;
  name: string;
  mechanism: string;
  protection_target: string;
  research_basis: string;
  models: AttackFamilyModel[];
}

export interface ProcessResult {
  session_id: string;
  width: number;
  height: number;
  face_count: number;
  faces: FaceBox[];
  person_count: number;
  persons: FaceBox[];
  faces_message: string;
  protection_applied: boolean;
  protection: ProtectionInfo;
  provenance: ProvenanceInfo;
  sensitive: {
    regions: Array<{
      kind: string;
      x1: number;
      y1: number;
      x2: number;
      y2: number;
      content: string | null;
      sensitive: boolean;
      experimental: boolean;
      note: string;
    }>;
    qr_codes: unknown[];
    text_regions: unknown[];
    ocr_available: boolean;
    ocr_note: string | null;
    experimental: boolean;
    summary: string;
  };
  metadata: {
    source_had_exif: boolean;
    source_had_gps: boolean;
    source_had_xmp: boolean;
    removed: string[];
    output_format: string;
    note: string;
    source: Record<string, unknown>;
  };
  quality: {
    psnr_db: number;
    ssim: number;
    mse: number;
    mae: number;
    perturbation_l2: number;
    perturbation_linf: number;
    lpips: number | null;
  };
  robustness: {
    overall: "PASS" | "PARTIAL" | "FAIL";
    transforms: Record<string, RobustnessTransformResult>;
    base_distances: Record<string, number>;
    model_summary: Record<string, { mean_distance: number; base_distance: number; verdict: string }>;
    thresholds: { pass: number; partial: number; unit: string };
    note: string;
  } | null;
  perception: Perception | null;
  editing: EditingInfo;
  families?: {
    profile: string;
    description: string;
    families: AttackFamily[];
    flags: Record<string, unknown>;
  };
  profile?: string;
  vlm: { enabled: boolean; note: string };
  processing_time_ms: number;
  models: {
    device: string;
    models: Array<{
      id: string;
      display_name: string;
      kind: string;
      architecture: string;
      license_note: string;
      loaded: boolean;
      error: string | null;
      device: string | null;
      state: string;
    }>;
  };
  hardware: {
    device: string;
    cuda: boolean;
    gpu_name: string | null;
    note: string;
  };
  original_data_url: string;
  protected_data_url: string;
  output_format: string;
  messages: string[];
}

// SSE event payloads ------------------------------------------------------

export type StageEvent = { type: "stage"; stage: string; message: string; loading_models?: boolean };
export type StageDoneEvent = { type: "stage_done"; stage: string; message: string };
export type FacesEvent = {
  type: "faces";
  count: number;
  faces: FaceBox[];
  person_count: number;
  persons: FaceBox[];
  message: string;
};
export type ProgressEvent = {
  type: "progress";
  iteration: number;
  total: number;
  phase: "optimize" | "refine" | "editing";
  loss?: number;
  distances?: Record<string, number>;
  message?: string;
};
export type ResultEvent = { type: "result" } & ProcessResult;
export type DoneEvent = { type: "done"; ok: boolean; message: string };
export type ErrorEvent = { type: "error"; message: string };
export type PingEvent = { type: "ping" };

export type ProcessEvent =
  | StageEvent
  | StageDoneEvent
  | FacesEvent
  | ProgressEvent
  | ResultEvent
  | DoneEvent
  | ErrorEvent
  | PingEvent;
