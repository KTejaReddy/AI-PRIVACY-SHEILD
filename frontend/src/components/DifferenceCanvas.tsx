import { useEffect, useRef, useState } from "react";
import { AMPLIFICATION_STEPS, computeDifferencePixels, stepToAmplification } from "../utils/difference";

interface Props {
  originalSrc: string;
  protectedSrc: string;
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Failed to load image."));
    img.src = src;
  });
}

export function DifferenceCanvas({ originalSrc, protectedSrc }: Props) {
  const displayRef = useRef<HTMLCanvasElement>(null);
  const [step, setStep] = useState(1); // index into AMPLIFICATION_STEPS
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [orig, prot] = await Promise.all([loadImage(originalSrc), loadImage(protectedSrc)]);
        if (cancelled || !displayRef.current) return;
        const w = Math.min(orig.naturalWidth, 800);
        const h = Math.round((orig.naturalHeight / orig.naturalWidth) * w);

        const read = (img: HTMLImageElement): Uint8ClampedArray => {
          const canvas = document.createElement("canvas");
          canvas.width = w;
          canvas.height = h;
          const ctx = canvas.getContext("2d", { willReadFrequently: true });
          if (!ctx) throw new Error("Canvas not supported.");
          ctx.drawImage(img, 0, 0, w, h);
          return ctx.getImageData(0, 0, w, h).data;
        };

        const origPixels = read(orig);
        const protPixels = read(prot);
        const amplification = stepToAmplification(step);

        const canvas = displayRef.current;
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext("2d");
        if (!ctx) throw new Error("Canvas not supported.");
        const diff = computeDifferencePixels(origPixels, protPixels, amplification);
        ctx.putImageData(new ImageData(diff, w, h), 0, 0);
        setError(null);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : "Difference rendering failed.");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [originalSrc, protectedSrc, step]);

  return (
    <div className="difference-panel">
      {error ? (
        <p className="upload-error">{error}</p>
      ) : (
        <canvas ref={displayRef} className="difference-canvas" aria-label="Amplified difference visualization" />
      )}
      <div className="amplification-control">
        <span className="amp-label">Difference Amplification</span>
        <input
          type="range"
          min={0}
          max={AMPLIFICATION_STEPS.length - 1}
          step={1}
          value={step}
          onChange={(e) => setStep(Number(e.target.value))}
          aria-label="Difference amplification"
        />
        <div className="amp-scale" aria-hidden="true">
          {AMPLIFICATION_STEPS.map((value, i) => (
            <span key={value} className={i === step ? "active" : ""}>
              {value}×
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
