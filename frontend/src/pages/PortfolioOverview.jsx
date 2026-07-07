import { useEffect, useState } from "react";
import {
  PieChart, Pie, Cell, Tooltip, ResponsiveContainer, LineChart, Line, XAxis, YAxis,
  CartesianGrid, Legend,
} from "recharts";
import api from "../api.js";
import StatCard from "../components/StatCard.jsx";
import LoadingSkeleton from "../components/LoadingSkeleton.jsx";
import { tooltipStyle } from "../chartTheme.js";

const RAG_COLORS = { Green: "#22c55e", Amber: "#f59e0b", Red: "#ef4444" };
const SMA_COLORS = { Regular: "#22c55e", "SMA-0": "#eab308", "SMA-1": "#f59e0b", "SMA-2": "#ef4444", NPA: "#991b1b" };

export default function PortfolioOverview() {
  const [data, setData] = useState(null);

  useEffect(() => {
    api.get("/portfolio").then((res) => setData(res.data));
  }, []);

  if (!data) {
    return (
      <div>
        <Header />
        <LoadingSkeleton height={500} />
      </div>
    );
  }

  const smaTotal = data.sma_breakdown.reduce((s, d) => s + d.accounts, 0);
  const eclTotal = { "Stage 1": 0, "Stage 2": 0, "Stage 3": 0 };
  data.ecl_staging.forEach((e) => (eclTotal[e.ecl_stage] = e));

  return (
    <div>
      <Header />

      <div className="grid grid-4" style={{ marginBottom: 20 }}>
        <StatCard label="Total Portfolio Exposure" value={data.total_portfolio_exposure_cr} decimals={1} prefix="₹" suffix=" Cr" />
        <StatCard label="Accounts Monitored" value={data.total_accounts} />
        <StatCard label="Predicted Default Rate" value={data.predicted_default_rate_pct} decimals={2} suffix="%" />
        <StatCard label="Model AUC-ROC" value={data.model_auc_roc} decimals={4} sub={`${data.accounts_at_risk} accounts flagged High/Critical`} />
      </div>

      <div className="grid grid-2" style={{ marginBottom: 20 }}>
        <div className="card">
          <div className="section-title">Risk Distribution (RAG)</div>
          <ResponsiveContainer width="100%" height={260}>
            <PieChart>
              <Pie
                data={data.rag_distribution}
                dataKey="accounts"
                nameKey="rag_status"
                innerRadius={65}
                outerRadius={100}
                paddingAngle={3}
                label={(e) => `${e.rag_status}: ${e.accounts}`}
              >
                {data.rag_distribution.map((d) => (
                  <Cell key={d.rag_status} fill={RAG_COLORS[d.rag_status] || "#64748b"} />
                ))}
              </Pie>
              <Tooltip contentStyle={tooltipStyle} formatter={(v, n, p) => [`${v} accounts · ₹${p.payload.exposure_cr} Cr`, p.payload.rag_status]} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <div className="section-title">SMA Classification Breakdown</div>
          <div style={{ display: "flex", height: 36, borderRadius: 8, overflow: "hidden", marginBottom: 16 }}>
            {data.sma_breakdown.map((d) => (
              <div
                key={d.sma_classification}
                title={`${d.sma_classification}: ${d.accounts}`}
                style={{
                  width: `${(d.accounts / smaTotal) * 100}%`,
                  background: SMA_COLORS[d.sma_classification] || "#64748b",
                }}
              />
            ))}
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {data.sma_breakdown.map((d) => (
              <div key={d.sma_classification} style={{ display: "flex", justifyContent: "space-between", fontSize: 13 }}>
                <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span className="dot" style={{ background: SMA_COLORS[d.sma_classification] }} />
                  {d.sma_classification}
                </span>
                <span style={{ color: "var(--text-dim)" }}>
                  {d.accounts.toLocaleString("en-IN")} accounts · ₹{d.exposure_cr} Cr
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="grid grid-3" style={{ marginBottom: 20 }}>
        {["Stage 1", "Stage 2", "Stage 3"].map((stage) => {
          const d = data.ecl_staging.find((e) => e.ecl_stage === stage);
          const color = stage === "Stage 1" ? "green" : stage === "Stage 2" ? "amber" : "red";
          return (
            <div key={stage} className="card">
              <div className="label" style={{ marginBottom: 8 }}>
                {stage} {stage === "Stage 1" ? "(12-month ECL)" : stage === "Stage 2" ? "(Lifetime ECL)" : "(Impaired)"}
              </div>
              <div className="value" style={{ color: `var(--${color})` }}>{d ? d.accounts.toLocaleString("en-IN") : 0}</div>
              <div className="sub">₹{d ? d.exposure_cr : 0} Cr exposure</div>
            </div>
          );
        })}
      </div>

      <div className="card" style={{ marginBottom: 20 }}>
        <div className="section-title">Portfolio Behavioral Risk Trend (12 Months)</div>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={data.monthly_risk_trend}>
            <CartesianGrid strokeDasharray="3 3" stroke="#2c3c54" />
            <XAxis dataKey="month" tick={{ fill: "#94a3b8", fontSize: 11 }} tickFormatter={(m) => m.slice(0, 7)} />
            <YAxis tick={{ fill: "#94a3b8", fontSize: 11 }} />
            <Tooltip contentStyle={tooltipStyle} />
            <Legend />
            <Line type="monotone" dataKey="portfolio_risk_score" name="Portfolio Risk Score" stroke="#3b82f6" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="card">
        <div className="section-title">Risk Concentration — Loan Type × City Tier</div>
        <div style={{ overflowX: "auto" }}>
          <table>
            <thead>
              <tr>
                <th>Loan Type</th>
                <th>City Tier</th>
                <th>Accounts</th>
                <th>Exposure (₹ Cr)</th>
                <th>Avg PD</th>
              </tr>
            </thead>
            <tbody>
              {data.risk_concentration
                .sort((a, b) => b.avg_pd - a.avg_pd)
                .map((r, i) => (
                  <tr key={i}>
                    <td>{r.loan_type}</td>
                    <td>Tier {r.city_tier}</td>
                    <td>{r.accounts}</td>
                    <td>{r.exposure_cr}</td>
                    <td>
                      <span
                        style={{
                          color: r.avg_pd >= 0.1 ? "var(--red)" : r.avg_pd >= 0.05 ? "var(--amber)" : "var(--green)",
                          fontWeight: 600,
                        }}
                      >
                        {(r.avg_pd * 100).toFixed(2)}%
                      </span>
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function Header() {
  return (
    <div className="page-header">
      <h1>IDBI Early Warning System — Default Prediction Engine</h1>
      <p>Portfolio-wide probability-of-default monitoring across Personal, Home, MSME and Auto loans</p>
    </div>
  );
}
