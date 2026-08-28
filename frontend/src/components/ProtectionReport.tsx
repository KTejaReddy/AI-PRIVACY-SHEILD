import type { ProcessResult } from "../types";

const MODEL_NAMES: Record<string, string> = {
  facenet_vggface2: "FaceNet (VGGFace2)",
  facenet_casia: "FaceNet (CASIA-WebFace)",
  arcface_mbf: "ArcFace (MobileFaceNet)",
};

const TRANSFORM_LABELS: Record<string, string> = {
  jpeg_compression: "JPEG compression",
  resize: "Resize",
  crop: "Crop",
  brightness: "Brightness",
  contrast: "Contrast",
  reencode: "Re-encoding",
};

function fmtVal(v: number | null | undefined): string {
  return v === null || v === undefined ? "not tested" : v.toFixed(3);
}

function fmtChange(pct: number | null | undefined): string {
  if (pct === null || pct === undefined) return "—";
  return `${pct > 0 ? "+" : ""}${pct.toFixed(0)}%`;
}

function VerdictBadge({ verdict }: { verdict: string }) {
  const cls = verdict === "PASS" ? "pass" : verdict === "PARTIAL" ? "partial" : "fail";
  return <span className={`verdict ${cls}`}>{verdict}</span>;
}

interface Props {
  result: ProcessResult;
}

export function ProtectionReport({ result }: Props) {
  const q = result.quality;
  const meta = result.metadata;
  const prot = result.protection;
  const fam = result.families?.families ?? [];
  const prov = result.provenance;

  const protectionText = prot.applied
    ? "A single imperceptible perturbation was applied to your photo, designed to reduce its usefulness for common AI editing and face-reuse techniques."
    : "Protection could not be applied on this system (a required local model is missing). The original photo was not modified or stored.";

  const rows: Array<{ label: string; value: React.ReactNode }> = [
    {
      label: "Faces",
      value: (
        <>
          {result.face_count} detected —{" "}
          {prot.applied && result.face_count > 0
            ? "included in the protection objective"
            : result.face_count === 0
              ? "none — image-wide protection applied"
              : "protection unavailable"}
        </>
      ),
    },
    {
      label: "Persons",
      value:
        result.person_count > 0
          ? `${result.person_count} detected — person regions included in the protection mask`
          : "None detected",
    },
    { label: "Visual similarity (SSIM)", value: q.ssim.toFixed(3) },
    { label: "Visual similarity (PSNR)", value: `${q.psnr_db.toFixed(1)} dB` },
    {
      label: "Perturbation",
      value: `max ${(q.perturbation_linf / 255).toFixed(3)} (0–1 scale), L2 ${q.perturbation_l2.toFixed(1)}`,
    },
    {
      label: "Metadata",
      value: (
        <>
          {meta.removed.length > 0 ? meta.removed.join("; ") : "No metadata to remove"}
          {meta.source_had_gps ? " (GPS present in source)" : ""}
        </>
      ),
    },
    {
      label: "Sensitive regions",
      value: (
        <>
          {result.sensitive.summary}
          {result.sensitive.experimental ? " (some detections are experimental)" : ""}
          {!result.sensitive.ocr_available ? " OCR unavailable." : ""}
        </>
      ),
    },
  ];

  return (
    <div className="report">
      <p className="report-note report-summary">
        <strong>{protectionText}</strong> Protection is evaluated against the known and tested
        AI manipulation families on this machine — it does not guarantee protection against
        every AI system or future models.
      </p>

      <details className="technical-details">
        <summary>Technical details (research / developer view)</summary>

        <h3>Protection analysis</h3>
        <dl className="report-grid">
          {rows.map((row) => (
            <div key={row.label} className="report-row">
              <dt>{row.label}</dt>
              <dd>{row.value}</dd>
            </div>
          ))}
        </dl>

        {prov && (
          <div className="provenance-section">
            <h3>Provenance (C2PA)</h3>
            <p className="report-note">
              {prov.applied
                ? `Embedded a cryptographically signed provenance manifest into the protected file. ${prov.note}`
                : prov.note}
            </p>
          </div>
        )}

        {fam.length > 0 && (
          <div className="families-section">
            <h3>Targeted AI attack families</h3>
            <p className="report-note">
              One imperceptible perturbation was optimized against these known AI manipulation
              mechanisms simultaneously (research basis in parentheses). This is not a claim of
              universal protection.
            </p>
            <ul className="families-list">
              {fam.map((f) => (
                <li key={f.id}>
                  <strong>{f.name}</strong>
                  <span className="small"> — {f.mechanism}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {result.perception && (
          <div className="perception-section">
            <h3>AI Perception Test</h3>
            <p className="report-note">{result.perception.note}</p>
            <table className="perception-table">
              <thead>
                <tr>
                  <th>Target (tested AI)</th>
                  <th>Original</th>
                  <th>Protected</th>
                  <th>Change</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Face detector confidence (OpenCV SSD)</td>
                  <td>{fmtVal(result.perception.faces.before)}</td>
                  <td>{fmtVal(result.perception.faces.after)}</td>
                  <td>{fmtChange(result.perception.faces.change_pct)}</td>
                </tr>
                {result.perception.faces_mtcnn?.tested && (
                  <tr>
                    <td>Face detector confidence (MTCNN)</td>
                    <td>{fmtVal(result.perception.faces_mtcnn.before)}</td>
                    <td>{fmtVal(result.perception.faces_mtcnn.after)}</td>
                    <td>{fmtChange(result.perception.faces_mtcnn.change_pct)}</td>
                  </tr>
                )}
                <tr>
                  <td>Person detector weight (HOG)</td>
                  <td>{fmtVal(result.perception.persons.before)}</td>
                  <td>{fmtVal(result.perception.persons.after)}</td>
                  <td>{fmtChange(result.perception.persons.change_pct)}</td>
                </tr>
                {result.perception.persons_neural?.tested && (
                  <tr>
                    <td>Person detector score (Faster R-CNN)</td>
                    <td>{fmtVal(result.perception.persons_neural.before)}</td>
                    <td>{fmtVal(result.perception.persons_neural.after)}</td>
                    <td>{fmtChange(result.perception.persons_neural.change_pct)}</td>
                  </tr>
                )}
                {Object.entries(result.perception.embeddings).map(([id, emb]) => (
                  <tr key={id}>
                    <td>{emb.display_name} — embedding similarity</td>
                    <td>{fmtVal(emb.before)}</td>
                    <td>{fmtVal(emb.after)}</td>
                    <td>{fmtChange(emb.change_pct)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="report-note">
              Confidence = detector output (lower is less reliably detected). Similarity = cosine
              similarity of the face embedding to the original identity (lower = more disrupted).
              {!result.vlm.enabled ? ` ${result.vlm.note}` : ""}
            </p>
          </div>
        )}

        {result.robustness && (
          <div className="robustness-section">
            <h3>
              Transformation robustness <VerdictBadge verdict={result.robustness.overall} />
            </h3>
            <p className="report-note">{result.robustness.note}</p>
            <table className="robustness-table">
              <thead>
                <tr>
                  <th>Transformation</th>
                  <th>Result</th>
                  <th>Mean disruption</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(result.robustness.transforms).map(([key, info]) => (
                  <tr key={key}>
                    <td>{TRANSFORM_LABELS[key] ?? key}</td>
                    <td>
                      <VerdictBadge verdict={info.verdict} />
                    </td>
                    <td>{info.mean.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="report-note">
              Disruption = L2 distance between the original and protected face embeddings in
              normalized space (pass ≥ {result.robustness.thresholds.pass}, partial ≥{" "}
              {result.robustness.thresholds.partial}). Higher is more disrupted.
            </p>
          </div>
        )}

        <EditingProtectionPanel editing={result.editing} />

        <div className="models-section">
          <h3>Models used</h3>
          <p className="report-note">
            {result.models.models
              .filter((m) => m.loaded)
              .map((m) => MODEL_NAMES[m.id] ?? m.display_name)
              .join(" · ") || "None loaded"}
          </p>
          <p className="report-note small">
            Protection engine profile: <strong>{result.profile ?? "production"}</strong>. The
            full research benchmark (attack families, held-out models, transformations) runs via{" "}
            <code>python scripts/benchmark_protection.py</code>.
          </p>
        </div>
      </details>
    </div>
  );
}

const EDITOR_NAMES: Record<string, string> = {
  instruction: "InstructPix2Pix (held out)",
  inpainting: "Masked inpainting (SD1.5)",
  image2image: "Image-to-image (SD1.5)",
};

function fmtRel(v: number | null | undefined): string {
  if (v === null || v === undefined) return "n/a";
  return `${v > 0 ? "+" : ""}${v.toFixed(0)}%`;
}

function EditingProtectionPanel({ editing }: { editing: ProcessResult["editing"] }) {
  const prot = editing.protection;
  const bench = editing.benchmark;
  const agg = bench.aggregate;
  const robustness = editing.robustness ?? bench.robustness ?? [];
  return (
    <div className="editing-section">
      <h3>AI Editing Protection</h3>
      {prot.applied ? (
        <p className="report-note">
          <strong>Protection applied.</strong> {prot.note}
          {prot.loss_increase_pct !== undefined && (
            <span className="stat-chip">
              SD1.5 denoising error {prot.loss_increase_pct.toFixed(0)}%↑ (verified at full
              resolution {prot.verified_increase_pct?.toFixed(0) ?? "—"}%↑)
            </span>
          )}
        </p>
      ) : (
        <p className="report-note">{prot.note}</p>
      )}

      <h4>Editing benchmark (original vs protected)</h4>
      {!bench.available ? (
        <p className="report-note">{bench.note}</p>
      ) : (
        <>
          <p className="report-note">{bench.note}</p>
          <table className="editing-table">
            <thead>
              <tr>
                <th>Editor</th>
                <th>Task</th>
                <th>Mask</th>
                <th>Original</th>
                <th>Protected</th>
                <th>Abs Δ</th>
                <th>Rel Δ</th>
              </tr>
            </thead>
            <tbody>
              {bench.tasks?.map((t) => (
                <tr key={t.id + t.editor_type}>
                  <td>{EDITOR_NAMES[t.editor_type] ?? t.editor_type}</td>
                  <td>{t.name}</td>
                  <td className="small">{t.mask_kind ?? "—"}</td>
                  <td>{t.success_original.toFixed(3)}</td>
                  <td>{t.success_protected.toFixed(3)}</td>
                  <td className={t.absolute_change > 0 ? "reduction-good" : "reduction-weak"}>
                    {t.absolute_change > 0 ? "+" : ""}
                    {t.absolute_change.toFixed(3)}
                  </td>
                  <td className="small">{fmtRel(t.relative_change_pct)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {agg && (
            <p className="report-note">
              <strong>
                Mean edit success {agg.mean_original.toFixed(3)} → {agg.mean_protected.toFixed(3)}{" "}
                (absolute change {agg.mean_absolute_change > 0 ? "+" : ""}
                {agg.mean_absolute_change.toFixed(3)}
                {agg.mean_relative_change_pct !== null
                  ? `, relative ${fmtRel(agg.mean_relative_change_pct)}`
                  : ""}).
              </strong>{" "}
              {agg.tasks_reduced}/{agg.tasks_total} task/editor rows showed lower edit success on
              the protected image. Success = task-specific pixel metric (did the requested change
              actually happen, measured in the edited region) + auxiliary CLIP semantic
              alignment. Relative change is shown only when the original success is meaningful;
              the raw task metric and CLIP components are reported in the JSON.
            </p>
          )}
          {bench.settings && (
            <p className="report-note small">
              {bench.settings.resolution}px, {bench.settings.steps} steps, guidance{" "}
              {bench.settings.guidance_scale}, seed {bench.settings.seed} · success ={" "}
              {Math.round(bench.settings.task_metric_weight * 100)}% task metric +{" "}
              {Math.round(bench.settings.clip_weight * 100)}% CLIP
            </p>
          )}

          {robustness.length > 0 && (
            <>
              <h4>Edit success after transformations (protected image)</h4>
              <table className="editing-table">
                <thead>
                  <tr>
                    <th>Transformation</th>
                    <th>Original</th>
                    <th>Protected + transform</th>
                  </tr>
                </thead>
                <tbody>
                  {robustness.map((r) => (
                    <tr key={r.transform}>
                      <td>{TRANSFORM_LABELS[r.transform] ?? r.transform}</td>
                      {r.error ? (
                        <td colSpan={2} className="reduction-weak">
                          error
                        </td>
                      ) : (
                        <>
                          <td>{r.mean_success_original?.toFixed(3)}</td>
                          <td>{r.mean_success_after_transform?.toFixed(3)}</td>
                        </>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="report-note">
                Same editor and settings, but the protected image was first JPEG-compressed /
                resized etc. Edit success after the transform stays low if the protection is
                robust to common re-encodings.
              </p>
            </>
          )}
        </>
      )}
    </div>
  );
}
