import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import { LayoutDashboard, Upload, History, Shield, Activity } from 'lucide-react'
import HomePage    from './pages/HomePage.jsx'
import UploadPage  from './pages/UploadPage.jsx'
import HistoryPage from './pages/HistoryPage.jsx'
import AnalysisPage from './pages/AnalysisPage.jsx'

const NAV = [
  { to: '/',        icon: LayoutDashboard, label: 'Overview' },
  { to: '/upload',  icon: Upload,          label: 'Upload Video' },
  { to: '/history', icon: History,         label: 'History' },
]

export default function App() {
  return (
    <BrowserRouter>
      <div className="layout">
        {/* Sidebar */}
        <aside className="sidebar">
          <div className="sidebar-logo">
            <div className="logo-icon">T</div>
            <div className="logo-text">
              <div className="title">TFIF</div>
              <div className="sub">Forensic Intelligence</div>
            </div>
          </div>
          <nav className="sidebar-nav">
            {NAV.map(({ to, icon: Icon, label }) => (
              <NavLink key={to} to={to} end={to==='/'} className={({ isActive }) => `nav-item${isActive ? ' active' : ''}`}>
                <Icon size={17} className="nav-icon" />
                {label}
              </NavLink>
            ))}
          </nav>
          <div className="sidebar-bottom">
            <div style={{ display:'flex', alignItems:'center', gap:6, marginBottom:6 }}>
              <Shield size={13} color="var(--green)" />
              <span style={{ color:'var(--green)', fontWeight:600 }}>System Active</span>
            </div>
            TFIF v1.0 &bull; AI Crime Reconstruction
          </div>
        </aside>

        {/* Main */}
        <main className="main-content">
          <Routes>
            <Route path="/"          element={<HomePage />} />
            <Route path="/upload"    element={<UploadPage />} />
            <Route path="/history"   element={<HistoryPage />} />
            <Route path="/analysis/:id" element={<AnalysisPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}
