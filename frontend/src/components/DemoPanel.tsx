import type { ProcessResult } from "../types";

interface Props {
  result: ProcessResult;
}

/**
 * Developer / demo panel. Hidden from the normal UI; enabled with
 * ?demo=1 in the URL or the ai-privacy-shield-demo localStorage flag.
 */
export function DemoPanel({ result }: Props) {
  return (
    <details className="demo-panel" open>
      <summary>Developer / demo data</summary>
      <div className="demo-body">
        <section>
          <h4>Hardware</h4>
          <pre>{JSON.stringify(result.hardware, null, 2)}</pre>
        </section>
        <section>
          <h4>Timing & pipeline</h4>
          <pre>{JSON.stringify({ processing_time_ms: result.processing_time_ms }, null, 2)}</pre>
        </section>
        <section>
          <h4>Detections</h4>
          <pre>{JSON.stringify({ faces: result.faces, persons: result.persons }, null, 2)}</pre>
        </section>
        <section>
          <h4>AI Perception Test (before/after)</h4>
          <pre>{JSON.stringify(result.perception, null, 2)}</pre>
        </section>
        <section>
          <h4>Vision-language status</h4>
          <pre>{JSON.stringify(result.vlm, null, 2)}</pre>
        </section>
        <section>
          <h4>Quality metrics</h4>
          <pre>{JSON.stringify(result.quality, null, 2)}</pre>
        </section>
        <section>
          <h4>Protection (optimizer)</h4>
          <pre>{JSON.stringify(result.protection, null, 2)}</pre>
        </section>
        <section>
          <h4>Model registry</h4>
          <pre>{JSON.stringify(result.models, null, 2)}</pre>
        </section>
        {result.robustness && (
          <>
            <section>
              <h4>Base embedding distances (protected vs original, no transform)</h4>
              <pre>{JSON.stringify(result.robustness.base_distances, null, 2)}</pre>
            </section>
            <section>
              <h4>Per-model summary</h4>
              <pre>{JSON.stringify(result.robustness.model_summary, null, 2)}</pre>
            </section>
            <section>
              <h4>Transformation tests</h4>
              <pre>{JSON.stringify(result.robustness.transforms, null, 2)}</pre>
            </section>
          </>
        )}
      </div>
    </details>
  );
}

export function demoEnabled(): boolean {
  if (typeof window === "undefined") return false;
  const urlFlag = new URLSearchParams(window.location.search).get("demo");
  if (urlFlag !== null) return urlFlag === "1" || urlFlag === "true";
  try {
    return window.localStorage.getItem("ai-privacy-shield-demo") === "1";
  } catch {
    return false;
  }
}
