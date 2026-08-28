"""C2PA provenance layer (spec §18, §19, §32).

C2PA is *not* an AI blocker. It is a second, independent defense layer that
cryptographically binds a provenance manifest to the protected file:

  * who processed the image (AI Privacy Shield + version),
  * what operation was applied (protection + timestamp),
  * the content hash of the protected image,
  * the application/software identity.

Any subsequent edit invalidates the signature, so a viewer can tell the
protected file was produced by this application and has not been tampered
with since. Platforms may strip C2PA metadata — the adversarial perturbation
remains the primary protection layer.

Implementation notes
--------------------
* Uses the ``c2pa-python`` binding of the open-source `c2pa-rs` SDK (Apache-2.0
  / MIT). See https://github.com/contentauth/c2pa-rs.
* The manifest is self-signed with a locally generated Ed25519 keypair so the
  feature works offline and does not depend on any commercial CA. Real
  deployments should provide their own signing key via
  ``AIPS_C2PA_KEY`` / ``AIPS_C2PA_CERT`` (the self-signed key only proves
  *internal* consistency, not identity trust).
* ``AIPS_C2PA_ENABLED=0`` disables the layer; if the dependency is missing or
  signing fails for any reason the ORIGINAL bytes are returned untouched and
  an honest status is reported — we never silently drop the user's image.
"""

from __future__ import annotations

import datetime
import hashlib
import io
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from ..config import settings

logger = logging.getLogger(__name__)

# Self-signed leaf certificates must carry a C2PA-valid EKU; the c2pa-rs SDK
# accepts id-kp-documentSigning (RFC 5280) among others (spec §14.5.1).
_DOCUMENT_SIGNING_OID = "1.3.6.1.5.5.7.3.36"
_CLAIM_GENERATOR = "aips/1.0.0"

_keypair_lock = threading.Lock()
_keypair: tuple[str, str] | None = None  # (private_key_pem, cert_pem)


@dataclass
class ProvenanceResult:
    """Outcome of the C2PA step. ``signed_bytes`` may equal the input."""

    available: bool
    enabled: bool
    applied: bool
    signed_bytes: bytes
    note: str


def c2pa_available() -> bool:
    try:
        import c2pa  # noqa: PLC0415, F401

        return True
    except Exception:  # noqa: BLE001
        return False


def _load_external_key() -> tuple[str, str] | None:
    """Load a user-provided signing key/cert (PEM). Returns None if unset."""
    key_path = settings.C2PA_KEY_PATH
    cert_path = settings.C2PA_CERT_PATH
    if not key_path or not cert_path:
        return None
    try:
        return (
            Path(key_path).read_text(encoding="utf-8"),
            Path(cert_path).read_text(encoding="utf-8"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read AIPS_C2PA_KEY/CERT: %s", exc)
        return None


def _self_signed_keypair() -> tuple[str, str] | None:
    """Generate (or reuse) a local Ed25519 keypair + self-signed leaf cert.

    The keypair is cached under ``AIPS_C2PA_KEY_DIR`` so every protected
    image on this machine is signed by the same (locally generated) identity.
    """
    global _keypair  # noqa: PLW0603
    with _keypair_lock:
        if _keypair is not None:
            return _keypair
        try:
            from cryptography import x509  # noqa: PLC0415
            from cryptography.hazmat.primitives import serialization  # noqa: PLC0415
            from cryptography.hazmat.primitives.asymmetric import ed25519  # noqa: PLC0415
            from cryptography.x509.oid import NameOID  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            logger.warning("C2PA keypair generation unavailable: %s", exc)
            return None

        key_dir = settings.C2PA_KEY_DIR
        key_path = key_dir / "aips_ed25519_key.pem"
        cert_path = key_dir / "aips_ed25519_cert.pem"
        try:
            if key_path.exists() and cert_path.exists():
                _keypair = (
                    key_path.read_text(encoding="utf-8"),
                    cert_path.read_text(encoding="utf-8"),
                )
                return _keypair
        except Exception:  # noqa: BLE001
            pass

        try:
            key = ed25519.Ed25519PrivateKey.generate()
            subject = issuer = x509.Name(
                [
                    x509.NameAttribute(NameOID.ORGANIZATION_NAME, "AI Privacy Shield"),
                    x509.NameAttribute(NameOID.COMMON_NAME, "aips.local"),
                ]
            )
            now = datetime.datetime.now(datetime.timezone.utc)
            cert = (
                x509.CertificateBuilder()
                .subject_name(subject)
                .issuer_name(issuer)
                .public_key(key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(now - datetime.timedelta(days=1))
                .not_valid_after(now + datetime.timedelta(days=3650))
                .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
                .add_extension(
                    x509.KeyUsage(
                        digital_signature=True,
                        content_commitment=True,
                        key_encipherment=False,
                        data_encipherment=False,
                        key_agreement=False,
                        key_cert_sign=False,
                        crl_sign=False,
                        encipher_only=False,
                        decipher_only=False,
                    ),
                    critical=True,
                )
                .add_extension(
                    x509.ExtendedKeyUsage([x509.ObjectIdentifier(_DOCUMENT_SIGNING_OID)]),
                    critical=False,
                )
                .add_extension(
                    x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False
                )
                .add_extension(
                    x509.AuthorityKeyIdentifier.from_issuer_public_key(key.public_key()),
                    critical=False,
                )
                .sign(key, None)  # Ed25519: algorithm must be None
            )
            priv_pem = key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            ).decode()
            cert_pem = cert.public_bytes(serialization.Encoding.PEM).decode()
            key_dir.mkdir(parents=True, exist_ok=True)
            key_path.write_text(priv_pem, encoding="utf-8")
            cert_path.write_text(cert_pem, encoding="utf-8")
            _keypair = (priv_pem, cert_pem)
            logger.info("Generated local C2PA signing keypair under %s", key_dir)
            return _keypair
        except Exception as exc:  # noqa: BLE001
            logger.warning("C2PA keypair generation failed: %s", exc)
            return None


def _build_manifest_json(content_hash: str, width: int, height: int, output_format: str) -> dict:
    """C2PA manifest JSON describing the protection operation."""
    return {
        "claim_generator": _CLAIM_GENERATOR,
        "claim_generator_info": [
            {"name": "AI Privacy Shield", "version": "1.0.0"}
        ],
        "assertions": [
            {
                "label": "c2pa.actions",
                "data": {
                    "actions": [
                        {
                            "action": "c2pa.placed",
                            "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/algorithmic",
                        }
                    ]
                },
            },
            {
                "label": "org.aips.protection",
                "data": {
                    "protection": "adversarial-multi-family",
                    "format": output_format,
                    "content_sha256": content_hash,
                    "width": width,
                    "height": height,
                    "note": (
                        "Imperceptible multi-family adversarial perturbation; C2PA binds "
                        "provenance but is not an AI blocker."
                    ),
                },
            },
        ],
    }


def add_c2pa_manifest(png_bytes: bytes, *, width: int, height: int, output_format: str) -> ProvenanceResult:
    """Embed a self-signed C2PA manifest into the protected PNG.

    Always returns the input bytes on any failure (never drops the image) and
    reports the honest status.
    """
    if not settings.C2PA_ENABLED:
        return ProvenanceResult(
            available=c2pa_available(), enabled=False, applied=False,
            signed_bytes=png_bytes, note="C2PA provenance disabled by configuration.",
        )
    if not c2pa_available():
        return ProvenanceResult(
            available=False, enabled=True, applied=False,
            signed_bytes=png_bytes,
            note=(
                "C2PA provenance unavailable: the 'c2pa-python' package is not installed. "
                "pip install c2pa-python to enable it."
            ),
        )
    try:
        import c2pa  # noqa: PLC0415
        from c2pa import C2paSignerInfo, C2paSigningAlg  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return ProvenanceResult(
            available=False, enabled=True, applied=False,
            signed_bytes=png_bytes, note=f"C2PA import failed: {type(exc).__name__}.",
        )

    keypair = _load_external_key() or _self_signed_keypair()
    if keypair is None:
        return ProvenanceResult(
            available=True, enabled=True, applied=False,
            signed_bytes=png_bytes,
            note="C2PA signing key could not be created; manifest not embedded.",
        )

    content_hash = hashlib.sha256(png_bytes).hexdigest()
    try:
        priv_pem, cert_pem = keypair
        signer = c2pa.Signer.from_info(
            C2paSignerInfo(
                alg=C2paSigningAlg.ED25519,
                private_key=priv_pem.encode(),
                sign_cert=cert_pem.encode(),
                ta_url=None,
            )
        )
        manifest_json = _build_manifest_json(content_hash, width, height, output_format)
        builder = c2pa.Builder(manifest_json)
        dest = io.BytesIO()
        builder.sign(signer, "image/png", io.BytesIO(png_bytes), dest)
        signed = dest.getvalue()
        if not signed.startswith(b"\x89PNG"):
            raise RuntimeError("C2PA output is not a valid PNG")
        return ProvenanceResult(
            available=True, enabled=True, applied=True,
            signed_bytes=signed,
            note=(
                "C2PA provenance manifest embedded (self-signed; content hash "
                f"{content_hash[:12]}…)."
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("C2PA embedding failed: %s", exc)
        return ProvenanceResult(
            available=True, enabled=True, applied=False,
            signed_bytes=png_bytes,
            note=f"C2PA signing failed ({type(exc).__name__}); image returned unsigned.",
        )
