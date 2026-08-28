import type { FacesEvent, ProgressEvent, StageDoneEvent } from "../types";

export interface StageState {
  key: string;
  label: string;
  status: "pending" | "active" | "done" | "error";
  detail?: string;
}

const STAGES: Array<{ key: string; label: string }> = [
  { key: "analyze", label: "Analyzing image" },
  { key: "faces", label: "Detecting faces" },
  { key: "sensitive", label: "Analyzing sensitive regions" },
  { key: "protect", label: "Protecting photo against AI manipulation" },
  { key: "treat", label: "Treating sensitive regions" },
  { key: "test", label: "Testing protection" },
  { key: "editing_benchmark", label: "Running AI-editing benchmark" },
  { key: "finalize", label: "Sanitizing metadata" },
  { key: "cleanup", label: "Clearing temporary data" },
];

export function buildStageStates(
  activeStage: string | null,
  doneStages: Set<string>,
  faces: FacesEvent | null,
  progress: ProgressEvent | null,
  errorMessage: string | null,
): StageState[] {
  return STAGES.map(({ key, label }) => {
    let status: StageState["status"] = "pending";
    if (errorMessage) {
      status = "pending";
      if (doneStages.has(key)) status = "done";
      if (key === activeStage) status = "error";
    } else if (doneStages.has(key)) {
      status = "done";
    } else if (key === activeStage) {
      status = "active";
    }
    let detail: string | undefined;
    if (key === "faces" && faces) {
      const personPart =
        faces.person_count > 0 ? `, ${faces.person_count} person${faces.person_count === 1 ? "" : "s"}` : "";
      detail =
        faces.count > 0
          ? `${faces.count} face${faces.count === 1 ? "" : "s"} detected${personPart}`
          : "No face detected — no facial-identity protection applied";
    } else if (key === "protect" && progress) {
      const pct = Math.min(100, Math.round((progress.iteration / progress.total) * 100));
      detail = progress.phase === "refine" ? `Refining for additional models… ${pct}%` : `Optimizing… ${pct}%`;
    } else if ((key === "editing" || key === "editing_benchmark") && progress?.message) {
      detail = progress.message;
    }
    return { key, label, status, detail };
  });
}

interface Props {
  activeStage: string | null;
  doneStages: Set<string>;
  faces: FacesEvent | null;
  progress: ProgressEvent | null;
  errorMessage: string | null;
}

export function ProgressPanel({ activeStage, doneStages, faces, progress, errorMessage }: Props) {
  const stages = buildStageStates(activeStage, doneStages, faces, progress, errorMessage);
  return (
    <div className="progress-panel" aria-live="polite">
      <h2 className="panel-title">Processing</h2>
      <ol className="stage-list">
        {stages.map((stage) => (
          <li key={stage.key} className={`stage-row ${stage.status}`}>
            <span className="stage-icon" aria-hidden="true">
              {stage.status === "done" ? "✓" : stage.status === "error" ? "!" : stage.status === "active" ? "●" : "○"}
            </span>
            <span className="stage-label">
              {stage.label}
              {stage.detail && <span className="stage-detail"> — {stage.detail}</span>}
            </span>
          </li>
        ))}
      </ol>
    </div>
  );
}

export type { StageDoneEvent };
