interface Props {
  message: string | null;
  kind: "ok" | "error";
  onDismiss: () => void;
}

export function StatusBanner({ message, kind, onDismiss }: Props) {
  if (!message) return null;
  return (
    <div className={`status-banner ${kind}`} role="status">
      <span>{message}</span>
      <button className="banner-dismiss" onClick={onDismiss} aria-label="Dismiss">
        ×
      </button>
    </div>
  );
}
