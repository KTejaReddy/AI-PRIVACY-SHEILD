import { useMemo } from "react";
import type { ProcessResult } from "../types";
import { ActionButtons, type ActionOutcome } from "./ActionButtons";
import { DemoPanel, demoEnabled } from "./DemoPanel";
import { DifferenceCanvas } from "./DifferenceCanvas";
import { ProtectionReport } from "./ProtectionReport";

interface Props {
  result: ProcessResult;
  onActionDone: (outcome: ActionOutcome) => void;
  onNewImage: () => void;
}

export function ResultView({ result, onActionDone, onNewImage }: Props) {
  const showDemo = useMemo(() => demoEnabled(), []);
  const mime = result.output_format === "jpeg" ? "image/jpeg" : "image/png";

  return (
    <div className="result-view">
      <div className="comparison-grid">
        <figure className="compare-card">
          <figcaption>Original</figcaption>
          <img src={result.original_data_url} alt="Original uploaded photo" />
        </figure>
        <figure className="compare-card">
          <figcaption>Protected</figcaption>
          <img src={result.protected_data_url} alt="Protected photo" />
        </figure>
        <figure className="compare-card">
          <figcaption>Difference</figcaption>
          <DifferenceCanvas
            originalSrc={result.original_data_url}
            protectedSrc={result.protected_data_url}
          />
        </figure>
      </div>

      <div className="faces-banner">
        {result.protection_applied ? (
          <span>
            Your photo has been protected. The protected version looks the same to a human but
            is designed to be substantially less useful for AI editing and face-reuse tools.
          </span>
        ) : (
          <span>{result.protection.message || result.faces_message}</span>
        )}
      </div>

      <ProtectionReport result={result} />

      <div className="actions-block">
        <ActionButtons
          protectedDataUrl={result.protected_data_url}
          outputFormat={result.output_format}
          onDone={onActionDone}
          onNewImage={onNewImage}
          disabled={false}
        />
        <p className="actions-note">
          Protected image format: {mime.toUpperCase()}. Temporary data is cleared automatically
          after copy or download.
        </p>
      </div>

      {showDemo && <DemoPanel result={result} />}
    </div>
  );
}
