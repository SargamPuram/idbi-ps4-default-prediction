import { useEffect, useState } from "react";
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Legend } from "recharts";
import api from "../api.js";
import StatCard from "../components/StatCard.jsx";
import LoadingSkeleton from "../components/LoadingSkeleton.jsx";
import { tooltipStyle, axisTick, gridStroke } from "../chartTheme.js";

const STAGE_COLORS = { "Stage 1": "#22c55e", "Stage 2": "#f59e0b", "Stage 3": "#ef4444" };
const LOAN_COLORS = { "Personal Loan": "#3b82f6", "Home Loan": "#22c55e", "MSME Loan": "#f59e0b", "Auto Loan": "#a855f7" };

export default function EclView() {
  const [data, setData] = useState(null);

  useEffect(() => {
    api.get("/ecl-summary").then((res) => setData(res.data));
  }, []);

  if (!data) return <LoadingSkeleton height={600} />;

  const byLoanType = {};
  data.stage_by_loan_type.forEach((r) => {
    byLoanType[r.ecl_stage] = byLoanType[r.ecl_stage] || { ecl_stage: r.ecl_stage };
    byLoanType[r.ecl_stage][r.loan_type] = r.exposure_cr;
  });
  const stages = ["Stage 1", "Stage 2", "Stage 3"];
  const loanTypes = ["Personal Loan", "Home Loan", "MSME Loan", "Auto Loan"];

  return (
    <div>
      <div className="page-header">
        <h1>ECL Regulatory View</h1>
        <p>{data.framework}</p>
      </div>

      <div className="grid grid-3" style={{ marginBottom: 20 }}>
        <StatCard label="Total Exposure" value={data.total_exposure_cr} decimals={1} prefix="₹" suffix=" Cr" />
        <StatCard label="Total ECL Provision" value={data.total_provision_cr} decimals={2} prefix="₹" suffix=" Cr" />
        <StatCard label="Provision Coverage Ratio" value={data.provision_coverage_ratio_pct} decimals={3} suffix="%" />
      </div>

      <div className="grid grid-2" style={{ marginBottom: 20 }}>
        <div className="card">
          <div className="section-title">Stage Distribution</div>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie
                data={data.stage_summary}
                dataKey="accounts"
                nameKey="ecl_stage"
                innerRadius={70}
                outerRadius={110}
                paddingAngle={3}
                label={(e) => `${e.ecl_stage}: ${e.accounts}`}
              >
                {data.stage_summary.map((d) => (
                  <Cell key={d.ecl_stage} fill={STAGE_COLORS[d.ecl_stage]} />
                ))}
              </Pie>
              <Tooltip contentStyle={tooltipStyle} formatter={(v, n, p) => [`${v} accounts · ₹${p.payload.provision_cr} Cr provision`, p.payload.ecl_stage]} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <div className="section-title">Exposure by Stage × Loan Type (₹ Cr)</div>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={stages.map((s) => byLoanType[s] || { ecl_stage: s })}>
              <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
              <XAxis dataKey="ecl_stage" tick={axisTick} />
              <YAxis tick={axisTick} />
              <Tooltip contentStyle={tooltipStyle} />
              <Legend />
              {loanTypes.map((lt) => (
                <Bar key={lt} dataKey={lt} stackId="a" fill={LOAN_COLORS[lt]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 20 }}>
        <div className="section-title">Stage Summary & Provisioning</div>
        <table>
          <thead>
            <tr><th>ECL Stage</th><th>Accounts</th><th>Exposure (₹ Cr)</th><th>Provision (₹ Cr)</th><th>Coverage %</th></tr>
          </thead>
          <tbody>
            {data.stage_summary.map((s) => (
              <tr key={s.ecl_stage}>
                <td><span className="badge" style={{ background: `${STAGE_COLORS[s.ecl_stage]}22`, color: STAGE_COLORS[s.ecl_stage] }}>{s.ecl_stage}</span></td>
                <td>{s.accounts}</td>
                <td>{s.exposure_cr}</td>
                <td>{s.provision_cr}</td>
                <td>{((s.provision_cr / s.exposure_cr) * 100).toFixed(2)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card">
        <div className="section-title">SMA Stage Transition Matrix (Month-over-Month)</div>
        <p style={{ fontSize: 12, color: "var(--text-dim)", marginTop: -6 }}>
          Probability an account in row-state transitions to column-state the following month.
        </p>
        <div style={{ overflowX: "auto" }}>
          <TransitionMatrix matrix={data.stage_transition_matrix} />
        </div>
      </div>
    </div>
  );
}

function TransitionMatrix({ matrix }) {
  const labels = ["Regular", "SMA-0", "SMA-1", "SMA-2", "NPA"];
  return (
    <table>
      <thead>
        <tr>
          <th>From \ To</th>
          {labels.map((l) => <th key={l}>{l}</th>)}
        </tr>
      </thead>
      <tbody>
        {labels.map((row) => (
          <tr key={row}>
            <td style={{ fontWeight: 600 }}>{row}</td>
            {labels.map((col) => {
              const p = matrix[row]?.[col] ?? 0;
              return (
                <td key={col} style={{ background: `rgba(59,130,246,${p})`, textAlign: "center" }}>
                  {(p * 100).toFixed(1)}%
                </td>
              );
            })}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
