import { useEffect, useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell,
} from "recharts";
import api from "../api.js";
import LoadingSkeleton from "../components/LoadingSkeleton.jsx";
import { tooltipStyle, axisTick, gridStroke } from "../chartTheme.js";

export default function ModelPerformance() {
  const [data, setData] = useState(null);

  useEffect(() => {
    api.get("/model/performance").then((res) => setData(res.data));
  }, []);

  if (!data) return <LoadingSkeleton height={600} />;

  const ensemble = data.model_comparison.find((m) => m.model === "Stacking Ensemble");
  const maxCm = Math.max(...data.confusion_matrix.matrix.flat());

  return (
    <div>
      <div className="page-header">
        <h1>Model Performance</h1>
        <p>Stacking ensemble (XGBoost + LightGBM + Logistic Regression) — validated on stratified and out-of-time holdouts</p>
      </div>

      <div className="grid grid-2" style={{ marginBottom: 20 }}>
        <div className="card">
          <div className="section-title">ROC Curve — AUC {ensemble?.auc_roc}</div>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={data.roc_curve}>
              <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
              <XAxis dataKey="fpr" type="number" domain={[0, 1]} tick={axisTick} label={{ value: "False Positive Rate", position: "insideBottom", offset: -5, fill: "#64748b", fontSize: 11 }} />
              <YAxis dataKey="tpr" type="number" domain={[0, 1]} tick={axisTick} />
              <Tooltip contentStyle={tooltipStyle} />
              <Line type="monotone" dataKey="tpr" stroke="#3b82f6" strokeWidth={2} dot={false} name="TPR" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <div className="section-title">Precision-Recall Curve — PR-AUC {ensemble?.pr_auc}</div>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={data.pr_curve}>
              <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
              <XAxis dataKey="recall" type="number" domain={[0, 1]} tick={axisTick} label={{ value: "Recall", position: "insideBottom", offset: -5, fill: "#64748b", fontSize: 11 }} />
              <YAxis dataKey="precision" type="number" domain={[0, 1]} tick={axisTick} />
              <Tooltip contentStyle={tooltipStyle} />
              <Line type="monotone" dataKey="precision" stroke="#22c55e" strokeWidth={2} dot={false} name="Precision" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid grid-2" style={{ marginBottom: 20 }}>
        <div className="card">
          <div className="section-title">Confusion Matrix (Stratified Holdout)</div>
          <div style={{ display: "grid", gridTemplateColumns: "auto 1fr 1fr", gap: 4, maxWidth: 380 }}>
            <div />
            <CmLabel text="Pred: No Default" />
            <CmLabel text="Pred: Default" />
            <CmLabel text="Actual: No Default" rotate />
            <CmCell value={data.confusion_matrix.matrix[0][0]} max={maxCm} good />
            <CmCell value={data.confusion_matrix.matrix[0][1]} max={maxCm} />
            <CmLabel text="Actual: Default" rotate />
            <CmCell value={data.confusion_matrix.matrix[1][0]} max={maxCm} />
            <CmCell value={data.confusion_matrix.matrix[1][1]} max={maxCm} good />
          </div>
        </div>

        <div className="card">
          <div className="section-title">Out-of-Time Validation</div>
          <p style={{ fontSize: 12, color: "var(--text-dim)", marginTop: -6 }}>
            Trained on oldest 75% of accounts by loan vintage, tested on newest 25% — simulates real deployment drift.
          </p>
          <div className="grid grid-3" style={{ marginTop: 12 }}>
            <MiniStat label="AUC-ROC" value={data.out_of_time_validation.auc_roc} />
            <MiniStat label="PR-AUC" value={data.out_of_time_validation.pr_auc} />
            <MiniStat label="F1" value={data.out_of_time_validation.f1} />
          </div>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 20 }}>
        <div className="section-title">Model Comparison</div>
        <table>
          <thead>
            <tr>
              <th>Model</th><th>AUC-ROC</th><th>PR-AUC</th><th>Precision</th><th>Recall</th><th>F1</th>
            </tr>
          </thead>
          <tbody>
            {data.model_comparison.map((m) => (
              <tr key={m.model} style={{ cursor: "default" }}>
                <td style={{ fontWeight: m.model === "Stacking Ensemble" ? 700 : 400 }}>{m.model}</td>
                <td>{m.auc_roc}</td>
                <td>{m.pr_auc}</td>
                <td>{m.precision}</td>
                <td>{m.recall}</td>
                <td>{m.f1}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card">
        <div className="section-title">Feature Importance (Top 20 — SHAP mean |value|)</div>
        <ResponsiveContainer width="100%" height={480}>
          <BarChart data={data.feature_importance} layout="vertical" margin={{ left: 40 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} horizontal={false} />
            <XAxis type="number" tick={axisTick} />
            <YAxis type="category" dataKey="feature" width={170} tick={{ ...axisTick, fontSize: 10.5 }} />
            <Tooltip contentStyle={tooltipStyle} />
            <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
              {data.feature_importance.map((_, i) => (
                <Cell key={i} fill={`rgba(59, 130, 246, ${1 - i * 0.04})`} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function CmLabel({ text, rotate }) {
  return (
    <div style={{ fontSize: 11, color: "var(--text-dim)", display: "flex", alignItems: "center", padding: 6, textAlign: rotate ? "right" : "center", justifyContent: rotate ? "flex-end" : "center" }}>
      {text}
    </div>
  );
}

function CmCell({ value, max, good }) {
  const intensity = value / max;
  return (
    <div
      style={{
        background: good ? `rgba(34,197,94,${0.15 + intensity * 0.6})` : `rgba(239,68,68,${0.1 + intensity * 0.5})`,
        borderRadius: 8,
        padding: "18px 8px",
        textAlign: "center",
        fontSize: 20,
        fontWeight: 700,
      }}
    >
      {value}
    </div>
  );
}

function MiniStat({ label, value }) {
  return (
    <div style={{ textAlign: "center" }}>
      <div style={{ fontSize: 20, fontWeight: 700 }}>{value}</div>
      <div style={{ fontSize: 11, color: "var(--text-faint)" }}>{label}</div>
    </div>
  );
}
