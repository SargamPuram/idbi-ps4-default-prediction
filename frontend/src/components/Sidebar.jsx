import { NavLink } from "react-router-dom";

const links = [
  { to: "/", label: "Portfolio Overview", icon: "▦" },
  { to: "/alerts", label: "Early Warning Alerts", icon: "⚠" },
  { to: "/model", label: "Model Performance", icon: "≈" },
  { to: "/ecl", label: "ECL Regulatory View", icon: "⚖" },
];

export default function Sidebar() {
  return (
    <aside
      style={{
        width: 240,
        flexShrink: 0,
        borderRight: "1px solid var(--border)",
        background: "#0b1120",
        padding: "24px 16px",
      }}
    >
      <div style={{ padding: "0 8px 28px" }}>
        <div style={{ fontSize: 13, color: "var(--idbi-blue)", fontWeight: 700, letterSpacing: 1 }}>
          IDBI BANK
        </div>
        <div style={{ fontSize: 15, fontWeight: 700, marginTop: 4, lineHeight: 1.3 }}>
          Early Warning System
        </div>
        <div style={{ fontSize: 12, color: "var(--text-faint)", marginTop: 2 }}>
          Default Prediction Engine
        </div>
      </div>
      <nav style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        {links.map((l) => (
          <NavLink
            key={l.to}
            to={l.to}
            end={l.to === "/"}
            style={({ isActive }) => ({
              display: "flex",
              alignItems: "center",
              gap: 10,
              padding: "10px 12px",
              borderRadius: 8,
              fontSize: 13.5,
              fontWeight: 500,
              color: isActive ? "#fff" : "var(--text-dim)",
              background: isActive ? "var(--idbi-blue)" : "transparent",
              transition: "background 0.15s",
            })}
          >
            <span>{l.icon}</span>
            {l.label}
          </NavLink>
        ))}
      </nav>
    </aside>
  );
}
