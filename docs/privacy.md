# Privacy

## What this software does and does not guarantee

**Does:** keeps processing on the user's own machine, never stores images in a
database, deletes its temporary copies, and never sends image data to any
third-party AI service.

**Does not:** claim "100% private" or "no AI can ever recognize this image."
Those are impossible to guarantee — a determined attacker with a model the
system was never tested against is outside the threat model. The protection
claim is always phrased against **tested models and transformations**.

## Where image data lives

| Stage | Location | Lifetime |
| --- | --- | --- |
| Browser object URL / blob | browser memory | revoked by `cleanup.ts` after copy/download / start-new |
| Upload staging | `backend/.tmp/sessions/<id>/` | deleted when the pipeline finishes, by `POST /api/cleanup`, or by the janitor (30 min TTL) |
| Model input tensors | GPU/CPU memory | freed after each stage |
| Protected result | returned over localhost as data URLs | cleared from React state after transfer |
| Anywhere else | — | never |

## Copy

The browser Clipboard API copies the image into the **operating system's
clipboard**. The application cannot delete an image from the OS clipboard, and
does not claim to. What the application guarantees is:

> "Application-owned temporary image data is cleared after copy."

After a successful copy, the app revokes object URLs, releases blobs and
buffers, and resets its state.

## Download

The browser writes the file to the user's device. The application cannot
delete a file after the user has downloaded it, and does not claim to. What
the application guarantees is:

> "Temporary copies held by this application are automatically cleared after
> download."

## Backend rules (enforced by design)

- No database is created for images; there is no image table anywhere.
- Temporary files use random server-generated session ids — no user-supplied
  filenames, no path traversal.
- Image contents are never logged; logs contain only session ids, dimensions
  and stage names.
- No analytics; no third-party HTTP calls from the processing path.
- Upload validation is content-based (magic bytes), never extension-based.
- The janitor sweeps stale sessions even if the browser crashes mid-stream.

## Sensitive content

- Faces are the primary protection target (multi-model adversarial
  perturbation).
- QR/barcode regions and OCR-detected text with PII-like patterns (phone
  numbers, emails, ID-like strings, addresses) are flagged and blurred in the
  output where the detector is confident.
- OCR is explicitly labeled **experimental** in the UI: it can miss text or
  misread it, and it does not claim to find every kind of personal
  information.

## Metadata

Every output image is re-encoded with EXIF, GPS, XMP, IPTC and all other
metadata removed, regardless of what the source contained. The report shows
what was found and removed.

## Provenance (C2PA)

By default the protected PNG also embeds a **C2PA provenance manifest** — a
cryptographically signed record of the protection operation, timestamp and
content hash (see `docs/consolidation.md` §5). Two important caveats:

- C2PA is **not** an AI blocker and never contains user data: the manifest
  names the application and operation only. If a platform strips C2PA
  metadata, the adversarial perturbation is still the primary layer.
- The bundled key is a locally generated self-signed identity that proves the
  file came from this application and has not been tampered with since — it
  does not assert who the signer is. Deployments that need verifiable-issuer
  provenance should configure their own signing certificate via
  `AIPS_C2PA_KEY` / `AIPS_C2PA_CERT` (disable with `AIPS_C2PA_ENABLED=0`).
