import { describe, expect, it, vi } from "vitest";
import { MAX_UPLOAD_BYTES, sniffImageFormat, validateImageFile } from "./imageValidation";

const PNG = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 1, 2, 3, 4]);
const JPEG = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 1, 2, 3, 4, 5, 6, 7, 8]);
const WEBP = new Uint8Array([0x52, 0x49, 0x46, 0x46, 1, 2, 3, 4, 0x57, 0x45, 0x42, 0x50]);
const TEXT = new TextEncoder().encode("hello, this is not an image");

describe("sniffImageFormat", () => {
  it("detects PNG from magic bytes", () => {
    expect(sniffImageFormat(PNG)).toBe("PNG");
  });
  it("detects JPEG from magic bytes", () => {
    expect(sniffImageFormat(JPEG)).toBe("JPEG");
  });
  it("detects WebP only when the RIFF tag is WEBP", () => {
    expect(sniffImageFormat(WEBP)).toBe("WebP");
    const notWebp = new Uint8Array(WEBP);
    notWebp[8] = 0x58; // not 'W'
    expect(sniffImageFormat(notWebp)).toBeNull();
  });
  it("rejects arbitrary text", () => {
    expect(sniffImageFormat(TEXT)).toBeNull();
  });
  it("rejects tiny buffers", () => {
    expect(sniffImageFormat(new Uint8Array([0x89]))).toBeNull();
  });
});

describe("validateImageFile", () => {
  function makeFile(bytes: Uint8Array, name: string): File {
    const blob = new Blob([bytes as unknown as BlobPart], { type: "application/octet-stream" });
    return new File([blob], name, { type: "application/octet-stream", lastModified: Date.now() });
  }

  it("accepts a valid PNG even with a wrong extension", async () => {
    const file = makeFile(PNG, "photo.txt");
    const result = await validateImageFile(file);
    expect(result.ok).toBe(true);
    expect(result.format).toBe("PNG");
  });

  it("rejects non-image content", async () => {
    const file = makeFile(TEXT, "fake.png");
    const result = await validateImageFile(file);
    expect(result.ok).toBe(false);
    expect(result.error).toMatch(/Unsupported or unrecognized/);
  });

  it("rejects empty files", async () => {
    const file = makeFile(new Uint8Array(0), "empty.png");
    const result = await validateImageFile(file);
    expect(result.ok).toBe(false);
  });

  it("rejects oversized files", async () => {
    const file = makeFile(new Uint8Array(16), "big.png");
    vi.spyOn(file, "size", "get").mockReturnValue(MAX_UPLOAD_BYTES + 1);
    const result = await validateImageFile(file);
    expect(result.ok).toBe(false);
    expect(result.error).toMatch(/upload limit/);
  });
});
