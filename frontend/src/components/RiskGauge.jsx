import { useEffect, useState } from "react";

export default function RiskGauge({ score, size = 220 }) {
  const [animated, setAnimated] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => setAnimated(score), 100);
    return () => clearTimeout(t);
  }, [score]);

  const radius = size / 2 - 14;
  const cx = size / 2;
  const cy = size / 2;
  const circumference = Math.PI * radius;
  const offset = circumference * (1 - animated / 100);

  const color = score >= 50 ? "#ef4444" : score >= 25 ? "#f59e0b" : score >= 10 ? "#eab308" : "#22c55e";

  return (
    <div style={{ position: "relative", width: size, height: size / 2 + 30 }}>
      <svg width={size} height={size / 2 + 20} viewBox={`0 0 ${size} ${size / 2 + 20}`}>
        <path
          d={`M ${cx - radius} ${cy} A ${radius} ${radius} 0 0 1 ${cx + radius} ${cy}`}
          fill="none"
          stroke="#2c3c54"
          strokeWidth={14}
          strokeLinecap="round"
        />
        <path
          d={`M ${cx - radius} ${cy} A ${radius} ${radius} 0 0 1 ${cx + radius} ${cy}`}
          fill="none"
          stroke={color}
          strokeWidth={14}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          style={{ transition: "stroke-dashoffset 1s cubic-bezier(0.22, 1, 0.36, 1), stroke 0.6s" }}
        />
      </svg>
      <div style={{ position: "absolute", top: cy - 18, left: 0, width: size, textAlign: "center" }}>
        <div style={{ fontSize: 34, fontWeight: 700, color }}>{animated.toFixed(0)}</div>
        <div style={{ fontSize: 11, color: "var(--text-faint)", letterSpacing: 0.5 }}>RISK SCORE / 100</div>
      </div>
    </div>
  );
}
