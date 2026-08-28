import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanupManager } from "./cleanup";

describe("cleanupManager", () => {
  beforeEach(() => {
    cleanupManager.setSession(null);
  });

  afterEach(async () => {
    await cleanupManager.release();
    vi.restoreAllMocks();
  });

  it("revokes registered object URLs", async () => {
    const revoke = vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
    cleanupManager.registerObjectUrl("blob:one");
    cleanupManager.registerObjectUrl("blob:two");
    const stats = await cleanupManager.release();
    expect(revoke).toHaveBeenCalledWith("blob:one");
    expect(revoke).toHaveBeenCalledWith("blob:two");
    expect(stats.objectUrlsRevoked).toBe(2);
    // second release is idempotent
    const again = await cleanupManager.release();
    expect(again.objectUrlsRevoked).toBe(0);
  });

  it("ignores non-blob URLs", async () => {
    const revoke = vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
    cleanupManager.registerObjectUrl("data:image/png;base64,xxx");
    await cleanupManager.release();
    expect(revoke).not.toHaveBeenCalled();
  });

  it("asks the backend to clean the session", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true });
    vi.stubGlobal("fetch", fetchMock);
    cleanupManager.setSession("abc123");
    const stats = await cleanupManager.release();
    expect(fetchMock).toHaveBeenCalledWith("/api/cleanup/abc123", { method: "POST" });
    expect(stats.backendSessionCleaned).toBe(true);
    // session was cleared by release: a new release does not call the backend again
    await cleanupManager.release();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("does not fail when the backend is unreachable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("down")));
    cleanupManager.setSession("xyz");
    const stats = await cleanupManager.release();
    expect(stats.backendSessionCleaned).toBe(false);
  });

  it("releases synchronously for page unload", () => {
    const revoke = vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
    cleanupManager.registerObjectUrl("blob:unload");
    cleanupManager.releaseSync();
    expect(revoke).toHaveBeenCalledWith("blob:unload");
  });
});
