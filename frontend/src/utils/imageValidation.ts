// Client-side image validation. The backend re-validates everything (we never
// trust the client), but this gives instant feedback before an upload.

export const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;

const MAGIC: Array<[Uint8Array, string]> = [
  [new Uint8Array([0xff, 0xd8, 0xff]), "JPEG"],
  [new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]), "PNG"],
  [new Uint8Array([0x52, 0x49, 0x46, 0x46]), "WebP"],
  [new Uint8Array([0x47, 0x49, 0x46, 0x38]), "GIF"],
  [new Uint8Array([0x42, 0x4d]), "BMP"],
];

/** Sniff the image format from raw bytes (magic bytes, not the extension). */
export function sniffImageFormat(bytes: Uint8Array): string | null {
  for (const [magic, name] of MAGIC) {
    if (bytes.length >= magic.length && magic.every((b, i) => bytes[i] === b)) {
      if (name === "WebP") {
        // RIFF .... WEBP
        if (bytes.length >= 12) {
          const tag = String.fromCharCode(bytes[8], bytes[9], bytes[10], bytes[11]);
          return tag === "WEBP" ? name : null;
        }
      }
      return name;
    }
  }
  return null;
}

export interface ValidationResult {
  ok: boolean;
  format: string | null;
  error: string | null;
}

/** Validate a selected File before it is uploaded. */
export function validateImageFile(file: File): Promise<ValidationResult> {
  if (file.size === 0) {
    return Promise.resolve({ ok: false, format: null, error: "The selected file is empty." });
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    return Promise.resolve({
      ok: false,
      format: null,
      error: `Image exceeds the ${MAX_UPLOAD_BYTES / (1024 * 1024)} MB upload limit.`,
    });
  }
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onerror = () =>
      resolve({ ok: false, format: null, error: "The file could not be read." });
    reader.onload = () => {
      const bytes = new Uint8Array(reader.result as ArrayBuffer);
      const format = sniffImageFormat(bytes);
      if (!format) {
        resolve({
          ok: false,
          format: null,
          error: "Unsupported or unrecognized image format. Supported: JPEG, PNG, WebP, BMP, GIF.",
        });
        return;
      }
      resolve({ ok: true, format, error: null });
    };
    reader.readAsArrayBuffer(file);
  });
}
