import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
  LineChart, Line, Legend,
} from "recharts";
import api from "../api.js";
import LoadingSkeleton from "../components/LoadingSkeleton.jsx";
import RiskGauge from "../components/RiskGauge.jsx";
import { RagBadge, SmaBadge, EclBadge } from "../components/Badges.jsx";
import { tooltipStyle, axisTick, gridStroke } from "../chartTheme.js";

export default function AccountDetail() {
  const { id } = useParams();
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    setData(null);
    setError(null);
    api
      .get(`/account/${id}`)
      .then((res) => setData(res.data))
      .catch(() => setError("Account not found"));
  }, [id]);

  if (error) return <div className="card">{error}. <Link to="/alerts">Back to alerts</Link></div>;
  if (!data) return <LoadingSkeleton height={600} />;

  const shapData = [...data.top_risk_drivers].reverse().map((d) => ({
    ...d,
    abs: Math.abs(d.shap_contribution),
  }));

  return (
    <div>
      <div className="page-header">
        <h1>{data.borrower_name}</h1>
        <p>
          {data.account_id} · {data.loan_type} · ₹{Number(data.loan_amount).toLocaleString("en-IN")} · {data.tenure_months} months @ {data.interest_rate}%
        </p>
      </div>

      <div className="grid" style={{ gridTemplateColumns: "280px 1fr", marginBottom: 20 }}>
        <div className="card" style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
          <RiskGauge score={data.risk_score} />
          <div style={{ display: "flex", gap: 8, marginTop: 12, flexWrap: "wrap", justifyContent: "center" }}>
            <RagBadge status={data.rag_status} />
            <SmaBadge status={data.sma_classification} />
            <EclBadge stage={data.ecl_stage} />
          </div>
        </div>

        <div className="card">
          <div className="section-title">Borrower Profile</div>
          <div className="grid grid-3">
            <Field label="Age / Gender" value={`${data.age} / ${data.gender}`} />
            <Field label="City" value={data.city} />
            <Field label="Employment" value={data.employment_type} />
            <Field label="Industry / Sector" value={data.industry_sector} />
            <Field label="Annual Income" value={`₹${Number(data.annual_income).toLocaleString("en-IN")}`} />
            <Field label="Credit Score" value={data.credit_score} />
            <Field label="Disbursement Date" value={data.disbursement_date?.slice(0, 10)} />
            <Field label="Probability of Default" value={`${(data.probability_of_default * 100).toFixed(2)}%`} />
            <Field label="Est. Months to Default" value={data.estimated_months_to_default ? `${data.estimated_months_to_default} mo` : "—"} />
          </div>

          <div className={`action-card ${data.risk_category === "Critical" ? "critical" : data.risk_category === "High" ? "high" : ""}`} style={{ marginTop: 16 }}>
            <strong>Recommended Action:</strong> {data.recommended_action}
          </div>
        </div>
      </div>

      <div className="grid grid-2" style={{ marginBottom: 20 }}>
        <div className="card">
          <div className="section-title">Top Risk Drivers (SHAP)</div>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={shapData} layout="vertical" margin={{ left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} horizontal={false} />
              <XAxis type="number" tick={axisTick} />
              <YAxis type="category" dataKey="label" width={160} tick={{ ...axisTick, fontSize: 10.5 }} />
              <Tooltip
                contentStyle={tooltipStyle}
                formatter={(v, n, p) => [p.payload.explanation, "Driver"]}
              />
              <Bar dataKey="shap_contribution" radius={[0, 4, 4, 0]}>
                {shapData.map((d, i) => (
                  <Cell key={i} fill={d.shap_contribution > 0 ? "#ef4444" : "#3b82f6"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p style={{ fontSize: 11, color: "var(--text-faint)", marginTop: 8 }}>
            Red bars increase default risk, blue bars reduce it — from the ensemble's XGBoost surrogate explainer.
          </p>
        </div>

        <div className="card">
          <div className="section-title">Payment Behavior — 12 Month Heatmap</div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 8 }}>
            {data.payment_history_12m.map((m) => {
              const status = m.emi_bounce_flag ? "bounce" : m.days_past_due === 0 ? "ok" : m.days_past_due <= 30 ? "late" : "overdue";
              const color = { ok: "#22c55e", late: "#eab308", overdue: "#f59e0b", bounce: "#ef4444" }[status];
              return (
                <div
                  key={m.month}
                  title={`${m.month}: DPD ${m.days_past_due}${m.emi_bounce_flag ? ", EMI bounced" : ""}`}
                  style={{
                    background: color,
                    opacity: status === "ok" ? 0.35 : 0.9,
                    borderRadius: 6,
                    padding: "10px 4px",
                    textAlign: "center",
                    fontSize: 10,
                    color: "#0b1120",
                    fontWeight: 700,
                  }}
                >
                  {m.month.slice(2, 7)}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <div className="grid grid-2">
        <div className="card">
          <div className="section-title">Historical Risk Trend</div>
          <ResponsiveContainer width="100%" height={230}>
            <LineChart data={data.historical_risk_trend}>
              <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
              <XAxis dataKey="month" tick={axisTick} tickFormatter={(m) => m.slice(2, 7)} />
              <YAxis tick={axisTick} />
              <Tooltip contentStyle={tooltipStyle} />
              <Line type="monotone" dataKey="risk_indicator" stroke="#ef4444" strokeWidth={2} dot={false} name="Risk Indicator" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <div className="section-title">Balance & Utilization Trend</div>
          <ResponsiveContainer width="100%" height={230}>
            <LineChart data={data.utilization_balance_trend_12m}>
              <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
              <XAxis dataKey="month" tick={axisTick} tickFormatter={(m) => m.slice(2, 7)} />
              <YAxis yAxisId="left" tick={axisTick} />
              <YAxis yAxisId="right" orientation="right" tick={axisTick} />
              <Tooltip contentStyle={tooltipStyle} />
              <Legend />
              <Line yAxisId="left" type="monotone" dataKey="utilization_ratio" stroke="#3b82f6" strokeWidth={2} dot={false} name="Utilization" />
              <Line yAxisId="right" type="monotone" dataKey="monthly_balance_avg" stroke="#22c55e" strokeWidth={2} dot={false} name="Balance (₹)" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

function Field({ label, value }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ fontSize: 11, color: "var(--text-faint)", textTransform: "uppercase", letterSpacing: 0.4 }}>{label}</div>
      <div style={{ fontSize: 14, fontWeight: 600 }}>{value}</div>
    </div>
  );
}
