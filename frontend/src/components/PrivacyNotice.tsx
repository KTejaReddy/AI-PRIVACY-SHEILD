// Privacy indicators — worded to match what the software actually guarantees.

export function PrivacyNotice() {
  return (
    <div className="privacy-notice" role="note">
      <div className="privacy-badge">
        <span className="privacy-dot" aria-hidden="true" />
        <span>
          <strong>Local-first processing</strong> — images are processed on this machine by the
          bundled local engine. No third-party AI services are used.
        </span>
      </div>
      <div className="privacy-badge">
        <span className="privacy-dot muted" aria-hidden="true" />
        <span>
          <strong>Images are not permanently stored</strong> by this application. Temporary files
          are deleted after processing, and after copy or download.
        </span>
      </div>
    </div>
  );
}
