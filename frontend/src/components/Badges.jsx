const RAG_COLOR = { Green: "green", Amber: "amber", Red: "red" };
const RISK_COLOR = { Low: "green", Medium: "amber", High: "red", Critical: "red" };

export function RagBadge({ status }) {
  const c = RAG_COLOR[status] || "slate";
  return (
    <span className={`badge badge-${c}`}>
      <span className="dot" style={{ background: `var(--${c === "slate" ? "text-faint" : c})` }} />
      {status}
    </span>
  );
}

export function RiskBadge({ category }) {
  const c = RISK_COLOR[category] || "slate";
  return <span className={`badge badge-${c}`}>{category}</span>;
}

export function SmaBadge({ status }) {
  const map = { Regular: "green", "SMA-0": "amber", "SMA-1": "amber", "SMA-2": "red", NPA: "red" };
  const c = map[status] || "slate";
  return <span className={`badge badge-${c}`}>{status}</span>;
}

export function EclBadge({ stage }) {
  const map = { "Stage 1": "green", "Stage 2": "amber", "Stage 3": "red" };
  const c = map[stage] || "slate";
  return <span className={`badge badge-${c}`}>{stage}</span>;
}

export function RiskProgressBar({ score }) {
  const color = score >= 50 ? "var(--red)" : score >= 25 ? "var(--amber)" : "var(--green)";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 90 }}>
      <div className="progress-track" style={{ flex: 1 }}>
        <div className="progress-fill" style={{ width: `${score}%`, background: color }} />
      </div>
      <span style={{ fontSize: 12, color: "var(--text-dim)", width: 32 }}>{score.toFixed(0)}</span>
    </div>
  );
}
