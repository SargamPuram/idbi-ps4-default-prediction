import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import api from "../api.js";
import LoadingSkeleton from "../components/LoadingSkeleton.jsx";
import { RagBadge, SmaBadge, EclBadge, RiskProgressBar } from "../components/Badges.jsx";

export default function Alerts() {
  const [alerts, setAlerts] = useState(null);
  const [filters, setFilters] = useState({ loan_type: "", risk_category: "", sma_status: "", search: "" });
  const navigate = useNavigate();

  useEffect(() => {
    const params = {};
    Object.entries(filters).forEach(([k, v]) => v && (params[k] = v));
    params.limit = 50;
    api.get("/alerts", { params }).then((res) => setAlerts(res.data.alerts));
  }, [filters]);

  return (
    <div>
      <div className="page-header">
        <h1>Early Warning Alerts</h1>
        <p>Top high-risk accounts requiring proactive attention, ranked by probability of default</p>
      </div>

      <div className="filter-bar">
        <select value={filters.loan_type} onChange={(e) => setFilters({ ...filters, loan_type: e.target.value })}>
          <option value="">All Loan Types</option>
          {["Personal Loan", "Home Loan", "MSME Loan", "Auto Loan"].map((t) => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
        <select value={filters.risk_category} onChange={(e) => setFilters({ ...filters, risk_category: e.target.value })}>
          <option value="">All Risk Categories</option>
          {["Low", "Medium", "High", "Critical"].map((t) => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
        <select value={filters.sma_status} onChange={(e) => setFilters({ ...filters, sma_status: e.target.value })}>
          <option value="">All SMA Status</option>
          {["Regular", "SMA-0", "SMA-1", "SMA-2", "NPA"].map((t) => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
        <input
          type="text"
          placeholder="Search by name or account ID..."
          value={filters.search}
          onChange={(e) => setFilters({ ...filters, search: e.target.value })}
          style={{ minWidth: 240 }}
        />
      </div>

      <div className="card">
        {!alerts ? (
          <LoadingSkeleton height={400} />
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table>
              <thead>
                <tr>
                  <th>Account</th>
                  <th>Borrower</th>
                  <th>Loan Type</th>
                  <th>Outstanding (₹)</th>
                  <th>Risk Score</th>
                  <th>RAG</th>
                  <th>SMA</th>
                  <th>ECL Stage</th>
                  <th>Est. Default Month</th>
                  <th>Top Risk Factor</th>
                </tr>
              </thead>
              <tbody>
                {alerts.map((a) => (
                  <tr key={a.account_id} onClick={() => navigate(`/account/${a.account_id}`)}>
                    <td style={{ fontFamily: "monospace", fontSize: 12 }}>{a.account_id}</td>
                    <td>{a.borrower_name}</td>
                    <td>{a.loan_type}</td>
                    <td>{Number(a.outstanding_amount).toLocaleString("en-IN")}</td>
                    <td><RiskProgressBar score={a.risk_score} /></td>
                    <td><RagBadge status={a.rag_status} /></td>
                    <td><SmaBadge status={a.sma_classification} /></td>
                    <td><EclBadge stage={a.ecl_stage} /></td>
                    <td>{a.estimated_months_to_default ? `${a.estimated_months_to_default} mo` : "—"}</td>
                    <td style={{ maxWidth: 260, fontSize: 12, color: "var(--text-dim)" }}>{a.top_risk_factor}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {alerts.length === 0 && (
              <div style={{ padding: 40, textAlign: "center", color: "var(--text-faint)" }}>No accounts match these filters.</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
