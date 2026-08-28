import { useCallback, useEffect, useRef, useState } from "react";
import { type ActionOutcome } from "./components/ActionButtons";
import { PrivacyNotice } from "./components/PrivacyNotice";
import { ProgressPanel } from "./components/ProgressPanel";
import { ResultView } from "./components/ResultView";
import { StatusBanner } from "./components/StatusBanner";
import { UploadZone } from "./components/UploadZone";
import { ApiError, streamProcess, uploadImage } from "./services/api";
import { cleanupManager, registerPageUnloadCleanup } from "./services/cleanup";
import type {
  FacesEvent,
  ProcessEvent,
  ProcessResult,
  ProgressEvent,
} from "./types";

type Phase = "idle" | "uploading" | "processing" | "result";

interface Banner {
  message: string;
  kind: "ok" | "error";
}

export default function App() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [activeStage, setActiveStage] = useState<string | null>(null);
  const [doneStages, setDoneStages] = useState<Set<string>>(new Set());
  const [faces, setFaces] = useState<FacesEvent | null>(null);
  const [progress, setProgress] = useState<ProgressEvent | null>(null);
  const [result, setResult] = useState<ProcessResult | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [banner, setBanner] = useState<Banner | null>(null);

  const resultRef = useRef<ProcessResult | null>(null);

  // Register page-unload cleanup once.
  useEffect(() => {
    registerPageUnloadCleanup();
  }, []);

  const resetToIdle = useCallback(() => {
    setPhase("idle");
    setActiveStage(null);
    setDoneStages(new Set());
    setFaces(null);
    setProgress(null);
    setResult(null);
    resultRef.current = null;
    setErrorMessage(null);
  }, []);

  const handleNewImage = useCallback(async () => {
    await cleanupManager.release();
    resetToIdle();
  }, [resetToIdle]);

  const handleFile = useCallback(
    async (file: File) => {
      setBanner(null);
      setErrorMessage(null);
      setPhase("uploading");
      try {
        const upload = await uploadImage(file);
        cleanupManager.setSession(upload.session_id);
        setPhase("processing");
        setActiveStage("analyze");

        await streamProcess(upload.session_id, (event: ProcessEvent) => {
          switch (event.type) {
            case "stage":
              setActiveStage(event.stage);
              break;
            case "stage_done":
              setDoneStages((prev) => new Set(prev).add(event.stage));
              setActiveStage(null);
              break;
            case "faces":
              setFaces(event);
              break;
            case "progress":
              setProgress({ ...event });
              break;
            case "result": {
              const payload = { ...(event as unknown as ProcessResult) };
              resultRef.current = payload;
              setResult(payload);
              setPhase("result");
              break;
            }
            case "done":
              // the cleanup stage completes when the stream finishes
              setDoneStages((prev) => new Set(prev).add("cleanup"));
              setActiveStage(null);
              break;
            case "error":
              setErrorMessage(event.message);
              break;
            default:
              break;
          }
        });
      } catch (err) {
        const message =
          err instanceof ApiError
            ? err.message
            : "The local processing engine could not be reached. Make sure the backend is running.";
        setErrorMessage(message);
      } finally {
        setActiveStage(null);
      }
    },
    [],
  );

  const handleActionDone = useCallback(
    (outcome: ActionOutcome) => {
      setBanner({ message: outcome.message, kind: outcome.ok ? "ok" : "error" });
      // Whether the action succeeded or the browser lacks support, the session
      // is over: release everything and return to the landing screen.
      void (async () => {
        await cleanupManager.release();
        resetToIdle();
      })();
    },
    [resetToIdle],
  );

  const processing = phase === "uploading" || phase === "processing";
  const showError = errorMessage !== null && phase !== "result";

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            🛡️
          </span>
          <div>
            <h1>AI Privacy Shield</h1>
            <p className="tagline">
              Imperceptible protection that reduces unauthorized AI image editing and
              synthetic-media generation while keeping your photo visually normal.
            </p>
          </div>
        </div>
        <div className="max-privacy-badge">MAXIMUM PRIVACY</div>
      </header>

      <main className="app-main">
        {phase === "idle" && (
          <>
            <PrivacyNotice />
            <UploadZone onFile={(file) => void handleFile(file)} disabled={processing} />
          </>
        )}

        {processing && (
          <ProgressPanel
            activeStage={activeStage}
            doneStages={doneStages}
            faces={faces}
            progress={progress}
            errorMessage={errorMessage}
          />
        )}

        {phase === "result" && result && (
          <ResultView
            result={result}
            onActionDone={handleActionDone}
            onNewImage={() => void handleNewImage()}
          />
        )}

        {showError && (
          <div className="error-panel" role="alert">
            <p>{errorMessage}</p>
            <button className="btn ghost" onClick={() => void handleNewImage()}>
              Start Over
            </button>
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>
          Protection is evaluated against the configured local models and transformations — it
          does not guarantee protection against every AI system.
        </p>
      </footer>

      <StatusBanner
        message={banner?.message ?? null}
        kind={banner?.kind ?? "ok"}
        onDismiss={() => setBanner(null)}
      />
    </div>
  );
}
