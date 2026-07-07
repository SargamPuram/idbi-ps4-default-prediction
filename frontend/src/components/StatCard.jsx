import AnimatedNumber from "./AnimatedNumber.jsx";

export default function StatCard({ label, value, decimals = 0, prefix = "", suffix = "", sub }) {
  return (
    <div className="card stat-card">
      <div className="label">{label}</div>
      <div className="value">
        <AnimatedNumber value={value} decimals={decimals} prefix={prefix} suffix={suffix} />
      </div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}
