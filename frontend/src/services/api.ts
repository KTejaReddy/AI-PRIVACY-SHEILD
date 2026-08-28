// Thin client for the local processing backend.
//
// The dev server proxies /api to the FastAPI backend (see vite.config.ts).

import type { HealthResponse, ProcessEvent, UploadResponse } from "../types";

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export async function fetchHealth(signal?: AbortSignal): Promise<HealthResponse> {
  const res = await fetch("/api/health", { signal });
  if (!res.ok) throw new ApiError("Local processing engine unavailable.", res.status);
  return res.json();
}

export async function uploadImage(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/upload", { method: "POST", body: form });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new ApiError(body?.error || "Upload failed.", res.status);
  }
  return body as UploadResponse;
}

export async function requestCleanup(sessionId: string): Promise<void> {
  try {
    await fetch(`/api/cleanup/${encodeURIComponent(sessionId)}`, { method: "POST" });
  } catch {
    // best-effort
  }
}

/**
 * Open the SSE processing stream and invoke `onEvent` for every event.
 * Resolves when the stream ends (done/error event or disconnect).
 */
export function streamProcess(
  sessionId: string,
  onEvent: (event: ProcessEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  return new Promise((resolve) => {
    const source = new EventSource(`/api/process/${encodeURIComponent(sessionId)}`);
    let settled = false;

    const finish = () => {
      if (settled) return;
      settled = true;
      source.close();
      resolve();
    };

    source.onmessage = (msg) => {
      let data: ProcessEvent;
      try {
        data = JSON.parse(msg.data) as ProcessEvent;
      } catch {
        return;
      }
      onEvent(data);
      if (data.type === "done" || data.type === "error") {
        finish();
      }
    };
    source.onerror = () => {
      // The backend closes the stream after done/error; treat an error here as
      // a disconnect (the generator already emitted what it had).
      finish();
    };
    signal?.addEventListener("abort", () => finish(), { once: true });
  });
}
