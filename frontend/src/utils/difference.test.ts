import { describe, expect, it } from "vitest";
import { AMPLIFICATION_STEPS, computeDifferencePixels, stepToAmplification } from "./difference";

function rgba(...pixels: number[]): Uint8ClampedArray {
  return new Uint8ClampedArray(pixels);
}

describe("computeDifferencePixels", () => {
  it("produces black output for identical images", () => {
    const a = rgba(10, 20, 30, 255, 200, 100, 50, 255);
    const out = computeDifferencePixels(a, rgba(...a), 10);
    expect(Array.from(out)).toEqual([0, 0, 0, 255, 0, 0, 0, 255]);
  });

  it("computes amplified per-channel absolute differences", () => {
    const orig = rgba(100, 100, 100, 255);
    const prot = rgba(110, 95, 120, 255); // +10, -5, +20
    const out = computeDifferencePixels(orig, prot, 2);
    expect(Array.from(out)).toEqual([20, 10, 40, 255]);
  });

  it("clips at 255", () => {
    const orig = rgba(0, 0, 0, 255);
    const prot = rgba(255, 255, 255, 255);
    const out = computeDifferencePixels(orig, prot, 100);
    expect(Array.from(out)).toEqual([255, 255, 255, 255]);
  });

  it("keeps alpha opaque", () => {
    const out = computeDifferencePixels(rgba(0, 0, 0, 0), rgba(10, 10, 10, 0), 1);
    expect(out[3]).toBe(255);
  });

  it("throws on mismatched buffer sizes", () => {
    expect(() => computeDifferencePixels(rgba(1, 2, 3, 4), rgba(1, 2, 3, 4, 5, 6, 7, 8), 1)).toThrow(
      /same size/,
    );
  });
});

describe("stepToAmplification", () => {
  it("maps slider steps to the preset amplification values", () => {
    expect(AMPLIFICATION_STEPS).toEqual([1, 5, 10, 25, 50, 100]);
    expect(stepToAmplification(0)).toBe(1);
    expect(stepToAmplification(1)).toBe(5);
    expect(stepToAmplification(3)).toBe(25);
    expect(stepToAmplification(5)).toBe(100);
  });
  it("clamps out-of-range steps", () => {
    expect(stepToAmplification(-3)).toBe(1);
    expect(stepToAmplification(99)).toBe(100);
    expect(stepToAmplification(2.6)).toBe(25);
  });
});
