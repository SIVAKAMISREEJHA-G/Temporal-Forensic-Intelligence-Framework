import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell, ResponsiveContainer,
  PieChart, Pie, Legend
} from 'recharts'
import { Video, CheckCircle, TrendingUp, AlertTriangle, Clock } from 'lucide-react'

const API = '/api'

const CLASS_COLORS = {
  Normal:'#3fb950', Fighting:'#ffa657', Shooting:'#f85149',
  Explosion:'#f85149', Riot:'#ffa657', 'Car Accident':'#e3b341', Abuse:'#bc8cff'
}

const SEVERITY = { Normal:'low', 'Car Accident':'medium', Abuse:'high', Fighting:'high', Riot:'high', Shooting:'critical', Explosion:'critical' }

function StatCard({ icon: Icon, value, label, color, glow }) {
  return (
    <div className="stat-card fade-in">
      <div className="stat-icon" style={{ background:`rgba(${glow},.12)` }}>
        <Icon size={20} color={color} />
      </div>
      <div className="stat-value" style={{ color }}>{value}</div>
      <div className="stat-label">{label}</div>
      <div className="stat-glow" style={{ background:color, filter:`blur(20px)` }} />
    </div>
  )
}

export default function HomePage() {
  const [stats, setStats] = useState(null)
  const navigate = useNavigate()

  useEffect(() => {
    fetch(`${API}/dashboard/stats`)
      .then(r => r.json()).then(setStats).catch(() => {})
  }, [])

  if (!stats) return (
    <div style={{ display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', height:'60vh', gap:16 }}>
      <div className="spinner" style={{ width:40, height:40 }} />
      <span className="text-muted">Loading dashboard…</span>
    </div>
  )

  const classDist = Object.entries(stats.class_distribution || {}).map(([name, value]) => ({
    name, value, color: CLASS_COLORS[name] || '#58a6ff'
  }))

  const totalIncidents = Object.entries(stats.class_distribution || {})
    .filter(([k]) => k !== 'Normal')
    .reduce((a,[,v]) => a+v, 0)

  return (
    <div className="fade-in">
      <div className="page-header">
        <h1>🔬 Intelligence Overview</h1>
        <div className="subtitle">Real-time forensic analysis dashboard</div>
      </div>

      <div className="stats-grid">
        <StatCard icon={Video}         value={stats.total_videos}      label="Videos Analysed"   color="#58a6ff" glow="88,166,255" />
        <StatCard icon={CheckCircle}   value={stats.processed_videos}  label="Reports Generated" color="#3fb950" glow="63,185,80" />
        <StatCard icon={AlertTriangle} value={totalIncidents}           label="Incidents Detected" color="#f85149" glow="248,81,73" />
        <StatCard icon={TrendingUp}    value={`${(stats.avg_confidence*100).toFixed(1)}%`} label="Avg Confidence" color="#bc8cff" glow="188,140,255" />
      </div>

      <div className="grid-2">
        {/* Bar chart */}
        <div className="card">
          <div className="section-title">Incidents by Class</div>
          {classDist.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={classDist} margin={{ top:4, right:8, left:-18, bottom:4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,.05)" />
                <XAxis dataKey="name" tick={{ fill:'#8b949e', fontSize:10 }} angle={-20} textAnchor="end" height={40} />
                <YAxis tick={{ fill:'#8b949e', fontSize:10 }} />
                <Tooltip contentStyle={{ background:'#1c2128', border:'1px solid #30363d', borderRadius:8, color:'#e6edf3', fontSize:12 }} />
                <Bar dataKey="value" radius={[4,4,0,0]}>
                  {classDist.map((d,i) => <Cell key={i} fill={d.color} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : <p className="text-muted" style={{ fontSize:'.85rem', marginTop:8 }}>No predictions yet.</p>}
        </div>

        {/* Pie chart */}
        <div className="card">
          <div className="section-title">Class Distribution</div>
          {classDist.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={classDist} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} paddingAngle={3} label={({ name, percent }) => `${name} ${(percent*100).toFixed(0)}%`} labelLine={false}>
                  {classDist.map((d,i) => <Cell key={i} fill={d.color} />)}
                </Pie>
                <Tooltip contentStyle={{ background:'#1c2128', border:'1px solid #30363d', borderRadius:8, color:'#e6edf3', fontSize:12 }} />
              </PieChart>
            </ResponsiveContainer>
          ) : <p className="text-muted" style={{ fontSize:'.85rem', marginTop:8 }}>No predictions yet.</p>}
        </div>
      </div>

      {/* Recent uploads */}
      <div className="card mt-16">
        <div className="flex items-center justify-between mb-16">
          <div className="section-title" style={{ margin:0 }}>Recent Uploads</div>
          <button className="btn btn-outline" style={{ fontSize:'.78rem', padding:'5px 12px' }} onClick={() => navigate('/upload')}>
            + New Upload
          </button>
        </div>
        {stats.recent_uploads?.length > 0 ? (
          <table className="data-table">
            <thead>
              <tr>
                <th>#</th><th>File</th><th>Class</th><th>Confidence</th><th></th>
              </tr>
            </thead>
            <tbody>
              {stats.recent_uploads.map(r => (
                <tr key={r.id}>
                  <td className="text-mono" style={{ fontSize:'.72rem', color:'var(--text-muted)' }}>TFIF-{String(r.id).padStart(6,'0')}</td>
                  <td style={{ maxWidth:200, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>{r.orig_name || '—'}</td>
                  <td>{r.predicted_class ? (
                    <span className={`badge badge-${SEVERITY[r.predicted_class]||'medium'}`}>{r.predicted_class}</span>
                  ) : '—'}</td>
                  <td>{r.confidence ? `${(r.confidence*100).toFixed(1)}%` : '—'}</td>
                  <td><button className="btn btn-outline" style={{ fontSize:'.72rem', padding:'4px 10px' }} onClick={() => navigate(`/analysis/${r.id}`)}>View</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <div style={{ textAlign:'center', padding:'32px', color:'var(--text-muted)' }}>
            No videos analysed yet. <button className="btn btn-primary" style={{ marginLeft:12 }} onClick={()=>navigate('/upload')}>Upload First Video</button>
          </div>
        )}
      </div>
    </div>
  )
}
