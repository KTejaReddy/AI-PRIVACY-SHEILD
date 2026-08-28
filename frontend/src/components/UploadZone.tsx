import { useCallback, useRef, useState } from "react";
import { validateImageFile } from "../utils/imageValidation";

interface Props {
  onFile: (file: File) => void;
  disabled: boolean;
}

export function UploadZone({ onFile, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const handleFile = useCallback(
    async (file: File | undefined) => {
      if (!file || disabled) return;
      setError(null);
      setBusy(true);
      const result = await validateImageFile(file);
      setBusy(false);
      if (!result.ok) {
        setError(result.error);
        return;
      }
      onFile(file);
    },
    [disabled, onFile],
  );

  return (
    <div
      className={`upload-zone ${dragging ? "dragging" : ""} ${disabled ? "disabled" : ""}`}
      role="button"
      tabIndex={0}
      aria-label="Upload an image"
      onClick={() => !disabled && inputRef.current?.click()}
      onKeyDown={(e) => {
        if ((e.key === "Enter" || e.key === " ") && !disabled) inputRef.current?.click();
      }}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        const file = e.dataTransfer.files?.[0];
        void handleFile(file);
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp,image/bmp,image/gif"
        style={{ display: "none" }}
        onChange={(e) => void handleFile(e.target.files?.[0])}
      />
      <div className="upload-icon" aria-hidden="true">
        <svg viewBox="0 0 24 24" width="44" height="44" fill="none" stroke="currentColor" strokeWidth="1.5">
          <path d="M12 16V4m0 0L7 9m5-5l5 5" strokeLinecap="round" strokeLinejoin="round" />
          <path d="M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3" strokeLinecap="round" />
        </svg>
      </div>
      <p className="upload-title">{busy ? "Checking image…" : "Upload Image"}</p>
      <p className="upload-hint">Drag &amp; drop a photo, or click to browse</p>
      <p className="upload-formats">JPEG · PNG · WebP · BMP · GIF (max 25 MB)</p>
      {error && <p className="upload-error">{error}</p>}
    </div>
  );
}
