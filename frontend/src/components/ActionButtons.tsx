import { useCallback, useState } from "react";
import { cleanupManager } from "../services/cleanup";

export interface ActionOutcome {
  ok: boolean;
  message: string;
  /** True when the application cleared its temporary data (normal path). */
  cleanedUp: boolean;
}

function dataUrlToBlob(dataUrl: string): Blob {
  const [header, payload] = dataUrl.split(",");
  const mime = header.match(/data:(.*?);/)?.[1] ?? "image/png";
  const binary = atob(payload);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Blob([bytes], { type: mime });
}

export async function copyImage(dataUrl: string): Promise<ActionOutcome> {
  if (typeof navigator.clipboard === "undefined" || typeof ClipboardItem === "undefined") {
    return {
      ok: false,
      cleanedUp: false,
      message: "Copy is not supported by this browser. Please use Download.",
    };
  }
  const blob = dataUrlToBlob(dataUrl);
  await navigator.clipboard.write([new ClipboardItem({ [blob.type]: blob })]);
  // The OS clipboard now owns a copy; we can clear everything we hold.
  const stats = await cleanupManager.release();
  return {
    ok: true,
    cleanedUp: stats.backendSessionCleaned || true,
    message:
      "Protected image copied. Application-owned temporary image data is cleared after copy. " +
      "(The clipboard copy itself cannot be deleted by the website.)",
  };
}

export async function downloadImage(
  dataUrl: string,
  filename: string,
  anchor?: HTMLAnchorElement,
): Promise<ActionOutcome> {
  const blob = dataUrlToBlob(dataUrl);
  const url = URL.createObjectURL(blob);
  cleanupManager.registerObjectUrl(url);
  try {
    const a = anchor ?? document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
  } catch {
    // fall through: even if the click failed we still clean up below
  }
  await cleanupManager.release();
  return {
    ok: true,
    cleanedUp: true,
    message:
      "Download started. Temporary copies held by this application are automatically cleared after download. " +
      "(A file already saved to your device cannot be deleted by the website.)",
  };
}

interface Props {
  protectedDataUrl: string;
  outputFormat: string;
  onDone: (outcome: ActionOutcome) => void;
  onNewImage: () => void;
  disabled: boolean;
}

export function ActionButtons({ protectedDataUrl, outputFormat, onDone, onNewImage, disabled }: Props) {
  const [busy, setBusy] = useState<"copy" | "download" | null>(null);

  const handleCopy = useCallback(async () => {
    if (disabled || busy) return;
    setBusy("copy");
    const outcome = await copyImage(protectedDataUrl);
    setBusy(null);
    onDone(outcome);
  }, [protectedDataUrl, disabled, busy, onDone]);

  const handleDownload = useCallback(async () => {
    if (disabled || busy) return;
    setBusy("download");
    const ext = outputFormat === "jpeg" ? "jpg" : "png";
    const outcome = await downloadImage(protectedDataUrl, `ai-privacy-shield-protected.${ext}`);
    setBusy(null);
    onDone(outcome);
  }, [protectedDataUrl, outputFormat, disabled, busy, onDone]);

  return (
    <div className="action-row">
      <button className="btn primary" onClick={() => void handleCopy()} disabled={disabled || busy !== null}>
        {busy === "copy" ? "Copying…" : "Copy Image"}
      </button>
      <button className="btn" onClick={() => void handleDownload()} disabled={disabled || busy !== null}>
        {busy === "download" ? "Downloading…" : "Download Image"}
      </button>
      <button className="btn ghost" onClick={onNewImage} disabled={disabled}>
        Start New Image
      </button>
    </div>
  );
}
