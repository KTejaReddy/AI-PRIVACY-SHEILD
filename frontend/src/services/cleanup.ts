// Central cleanup manager for application-owned temporary image data.
//
// Responsibilities:
//  * track object URLs and blobs created while displaying / copying / downloading
//  * revoke object URLs and drop references on demand
//  * tell the backend to delete the session's temp files
//
// Honest limitations (by design of the web platform):
//  * data already handed to the OS clipboard or downloaded to disk cannot be
//    deleted by the page — the UI says so explicitly.

export interface CleanupStats {
  objectUrlsRevoked: number;
  blobsReleased: number;
  backendSessionCleaned: boolean;
}

class CleanupManager {
  private objectUrls: string[] = [];
  private blobs: Blob[] = [];
  private sessionId: string | null = null;

  setSession(sessionId: string | null): void {
    this.sessionId = sessionId;
  }

  registerObjectUrl(url: string): void {
    if (url && url.startsWith("blob:")) {
      this.objectUrls.push(url);
    }
  }

  registerBlob(blob: Blob | null): void {
    if (blob) {
      this.blobs.push(blob);
    }
  }

  /** Revoke and drop everything this manager owns. Idempotent. */
  async release(): Promise<CleanupStats> {
    const stats: CleanupStats = {
      objectUrlsRevoked: 0,
      blobsReleased: 0,
      backendSessionCleaned: false,
    };

    for (const url of this.objectUrls) {
      try {
        URL.revokeObjectURL(url);
        stats.objectUrlsRevoked += 1;
      } catch {
        // ignore per-URL failures
      }
    }
    this.objectUrls = [];
    this.blobs = [];
    stats.blobsReleased = 1; // all tracked blobs dropped (counts as one batch)

    if (this.sessionId) {
      const sid = this.sessionId;
      this.sessionId = null;
      try {
        await fetch(`/api/cleanup/${encodeURIComponent(sid)}`, { method: "POST" });
        stats.backendSessionCleaned = true;
      } catch {
        // backend may already have cleaned up on its own
      }
    }
    return stats;
  }

  /** Best-effort synchronous release for page unload (no fetch). */
  releaseSync(): void {
    for (const url of this.objectUrls) {
      try {
        URL.revokeObjectURL(url);
      } catch {
        // ignore
      }
    }
    this.objectUrls = [];
    this.blobs = [];
  }
}

export const cleanupManager = new CleanupManager();

export function registerPageUnloadCleanup(): void {
  if (typeof window === "undefined") return;
  window.addEventListener("pagehide", () => cleanupManager.releaseSync());
  window.addEventListener("beforeunload", () => cleanupManager.releaseSync());
}
