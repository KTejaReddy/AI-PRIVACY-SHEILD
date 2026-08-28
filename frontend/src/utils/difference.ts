// Difference visualization: amplified |protected - original|.
// Pure function so it is trivially testable and runs without touching the DOM.

export type RGBA = Uint8ClampedArray;
/** Buffer guaranteed to be backed by a plain ArrayBuffer (ImageData-ready). */
export type RGBABuffer = Uint8ClampedArray<ArrayBuffer>;

/**
 * Compute `amp * |a - b|` per channel, clipped to [0, 255].
 * Both inputs must have the same length (RGBA buffers).
 */
export function computeDifferencePixels(
  original: RGBA,
  protectedImage: RGBA,
  amplification: number,
): RGBABuffer {
  if (original.length !== protectedImage.length) {
    throw new Error("Image buffers must have the same size.");
  }
  const out = new Uint8ClampedArray(original.length);
  for (let i = 0; i < original.length; i += 4) {
    const dr = Math.abs(protectedImage[i] - original[i]) * amplification;
    const dg = Math.abs(protectedImage[i + 1] - original[i + 1]) * amplification;
    const db = Math.abs(protectedImage[i + 2] - original[i + 2]) * amplification;
    out[i] = Math.min(255, dr);
    out[i + 1] = Math.min(255, dg);
    out[i + 2] = Math.min(255, db);
    out[i + 3] = 255;
  }
  return out;
}

/** Standard amplification presets for the slider. */
export const AMPLIFICATION_STEPS = [1, 5, 10, 25, 50, 100] as const;

export function stepToAmplification(step: number): number {
  const clamped = Math.max(0, Math.min(AMPLIFICATION_STEPS.length - 1, Math.round(step)));
  return AMPLIFICATION_STEPS[clamped];
}
