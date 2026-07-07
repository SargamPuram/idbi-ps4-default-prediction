import { Route, Routes } from "react-router-dom";
import Sidebar from "./components/Sidebar.jsx";
import PortfolioOverview from "./pages/PortfolioOverview.jsx";
import Alerts from "./pages/Alerts.jsx";
import AccountDetail from "./pages/AccountDetail.jsx";
import ModelPerformance from "./pages/ModelPerformance.jsx";
import EclView from "./pages/EclView.jsx";

export default function App() {
  return (
    <div className="app-shell">
      <Sidebar />
      <main className="main-content">
        <Routes>
          <Route path="/" element={<PortfolioOverview />} />
          <Route path="/alerts" element={<Alerts />} />
          <Route path="/account/:id" element={<AccountDetail />} />
          <Route path="/model" element={<ModelPerformance />} />
          <Route path="/ecl" element={<EclView />} />
        </Routes>
      </main>
    </div>
  );
}
