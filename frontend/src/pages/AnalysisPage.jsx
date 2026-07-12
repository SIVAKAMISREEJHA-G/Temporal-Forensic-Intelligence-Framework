import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell, ResponsiveContainer } from 'recharts'
import { Download, ArrowLeft, Shield, Clock, Image as ImgIcon, FileText } from 'lucide-react'

const API = '/api'

const SEVERITY = { Normal:'low','Car Accident':'medium',Abuse:'high',Fighting:'high',Riot:'high',Shooting:'critical',Explosion:'critical' }
const ACTIVITY_COLOR = { Low:'var(--severity-low)', Medium:'var(--severity-medium)', High:'var(--severity-high)' }
const CLASS_COLORS = {
  Normal:'#3fb950', Fighting:'#ffa657', Shooting:'#f85149',
  Explosion:'#f85149', Riot:'#ffa657','Car Accident':'#e3b341', Abuse:'#bc8cff'
}

const CLASS_ICONS = {
  Normal:'🟢', Fighting:'🥊', Shooting:'🔫', Explosion:'💥', Riot:'🏳', 'Car Accident':'🚗', Abuse:'⚠️'
}

function ReportSection({ icon, title, content }) {
  const isArray = Array.isArray(content)
  return (
    <div className="report-section">
      <h3>{icon} {title}</h3>
      {isArray ? (
        <ul className="bullet-list content">
          {content.map((c,i) => <li key={i}>{c}</li>)}
        </ul>
      ) : (
        <div className="content">{content}</div>
      )}
    </div>
  )
}

function ConfBar({ label, value, max=1, color }) {
  const pct = Math.min(100, (value / max) * 100)
  return (
    <div className="conf-bar-item">
      <div className="conf-bar-label">
        <span className="cls">{CLASS_ICONS[label]} {label}</span>
        <span className="pct">{(value*100).toFixed(1)}%</span>
      </div>
      <div className="conf-bar-track">
        <div className="conf-bar-fill" style={{ width:`${pct}%`, background: color || CLASS_COLORS[label] || '#58a6ff' }} />
      </div>
    </div>
  )
}

export default function AnalysisPage() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [result, setResult]   = useState(null)
  const [report, setReport]   = useState(null)
  const [loading, setLoading] = useState(true)
  const [tab, setTab]         = useState('overview')

  useEffect(() => {
    Promise.all([
      fetch(`${API}/videos/${id}/result`).then(r => r.ok ? r.json() : null),
      fetch(`${API}/videos/${id}/report`).then(r => r.ok ? r.json() : null),
    ]).then(([res, rep]) => {
      setResult(res); setReport(rep); setLoading(false)
    }).catch(() => setLoading(false))
  }, [id])

  const downloadReport = () => window.open(`${API}/videos/${id}/report/download`, '_blank')

  if (loading) return (
    <div style={{ display:'flex',alignItems:'center',justifyContent:'center',height:'70vh',gap:16 }}>
      <div className="spinner" style={{ width:40,height:40 }} />
      <span className="text-muted">Loading analysis…</span>
    </div>
  )

  if (!result) return (
    <div style={{ textAlign:'center', padding:'60px' }}>
      <p className="text-muted" style={{ marginBottom:16 }}>No analysis data found for this video.</p>
      <button className="btn btn-outline" onClick={() => navigate(-1)}><ArrowLeft size={14}/> Back</button>
    </div>
  )

  const sev  = SEVERITY[result.predicted_class] || 'medium'
  const chartData = Object.entries(result.per_class_scores||{}).map(([name, value]) => ({ name, value }))
  const rep  = report?.report

  const TABS = [
    { key:'overview', label:'📊 Overview' },
    { key:'timeline', label:'⏱ Timeline' },
    { key:'evidence', label:'🖼 Evidence' },
    { key:'report',   label:'📄 Report' },
  ]

  return (
    <div className="fade-in">
      {/* Header */}
      <div className="flex items-center justify-between mb-16">
        <div style={{ display:'flex', alignItems:'center', gap:12 }}>
          <button className="btn btn-outline" style={{ padding:'6px 12px' }} onClick={() => navigate(-1)}>
            <ArrowLeft size={14} />
          </button>
          <div>
            <h1 style={{ fontSize:'1.3rem', fontWeight:700 }}>Case TFIF-{String(id).padStart(6,'0')}</h1>
            <div style={{ fontSize:'.78rem', color:'var(--text-muted)' }}>Analysis Report</div>
          </div>
        </div>
        <div style={{ display:'flex', gap:10, alignItems:'center' }}>
          <span className={`badge badge-${sev}`} style={{ fontSize:'.8rem', padding:'5px 14px' }}>
            <Shield size={12} /> {result.predicted_class}
          </span>
          <button className="btn btn-primary" onClick={downloadReport}>
            <Download size={14} /> Download PDF
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div style={{ display:'flex', gap:4, marginBottom:20, borderBottom:'1px solid var(--border)', paddingBottom:2 }}>
        {TABS.map(t => (
          <button key={t.key} onClick={() => setTab(t.key)}
            style={{ padding:'7px 16px', borderRadius:'8px 8px 0 0', border:'none', cursor:'pointer', fontSize:'.85rem', fontWeight:600, transition:'all .15s',
              background: tab===t.key ? 'var(--bg-card)' : 'transparent',
              color: tab===t.key ? 'var(--blue)' : 'var(--text-secondary)',
              borderBottom: tab===t.key ? '2px solid var(--blue)' : '2px solid transparent',
            }}>
            {t.label}
          </button>
        ))}
      </div>

      {/* ── Overview Tab ── */}
      {tab === 'overview' && (
        <div className="fade-in">
          <div className="grid-2">
            {/* Confidence bars */}
            <div className="card">
              <div className="section-title">Confidence Scores</div>
              <div style={{ display:'flex', alignItems:'center', gap:10, marginBottom:16, padding:'10px 14px', background:'var(--bg-card2)', borderRadius:10, border:'1px solid var(--border)' }}>
                <span style={{ fontSize:1.6+'rem' }}>{CLASS_ICONS[result.predicted_class]}</span>
                <div>
                  <div style={{ fontWeight:700, fontSize:'1.1rem' }}>{result.predicted_class}</div>
                  <div style={{ fontSize:'.8rem', color:'var(--text-secondary)' }}>
                    Confidence: <span style={{ color:'var(--blue)', fontWeight:600 }}>{(result.confidence*100).toFixed(1)}%</span>
                  </div>
                </div>
                <span className={`badge badge-${sev}`} style={{ marginLeft:'auto' }}>{sev.toUpperCase()}</span>
              </div>
              <div className="conf-bar-list">
                {chartData.sort((a,b) => b.value-a.value).map(({ name, value }) => (
                  <ConfBar key={name} label={name} value={value} />
                ))}
              </div>
            </div>

            {/* Bar chart */}
            <div className="card">
              <div className="section-title">Class Probability Chart</div>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={chartData} layout="vertical" margin={{ top:4, right:16, left:70, bottom:4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,.04)" horizontal={false} />
                  <XAxis type="number" domain={[0,1]} tickFormatter={v=>`${(v*100).toFixed(0)}%`} tick={{ fill:'#8b949e', fontSize:10 }} />
                  <YAxis type="category" dataKey="name" tick={{ fill:'#e6edf3', fontSize:10 }} width={70} />
                  <Tooltip formatter={v=>`${(v*100).toFixed(2)}%`} contentStyle={{ background:'#1c2128', border:'1px solid #30363d', borderRadius:8, color:'#e6edf3', fontSize:12 }} />
                  <Bar dataKey="value" radius={[0,4,4,0]}>
                    {chartData.map((d,i) => <Cell key={i} fill={CLASS_COLORS[d.name]||'#58a6ff'} />)}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      )}

      {/* ── Timeline Tab ── */}
      {tab === 'timeline' && (
        <div className="fade-in card">
          <div className="section-title">⏱ Temporal Event Timeline</div>
          <div className="timeline" style={{ marginTop:16 }}>
            {(result.timeline||[]).map((seg, i) => {
              const col = ACTIVITY_COLOR[seg.activity_level] || 'var(--blue)'
              return (
                <div key={i} className="timeline-item">
                  <div className="timeline-dot" style={{ background:col, borderColor:col }} />
                  <div className="time">{seg.time_start}s – {seg.time_end}s</div>
                  <div className="label">{seg.label}</div>
                  <div className="detail">
                    Activity Level: <span style={{ color:col, fontWeight:600 }}>{seg.activity_level}</span>
                    &nbsp;·&nbsp;Attention Score: <span className="text-mono" style={{ fontSize:'.75rem' }}>{seg.avg_attention}</span>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* ── Evidence Tab ── */}
      {tab === 'evidence' && (
        <div className="fade-in card">
          <div className="section-title flex items-center gap-12">
            <ImgIcon size={16} /> Key Evidence Frames
            <span style={{ fontWeight:400, fontSize:'.8rem', color:'var(--text-secondary)', marginLeft:8 }}>
              {(result.keyframes||[]).length} frames selected by attention weights
            </span>
          </div>
          <div className="evidence-grid" style={{ marginTop:16 }}>
            {(result.keyframes||[]).map((kf, i) => (
              <div key={i} className="evidence-card">
                <img
                  src={kf.url}
                  alt={`Evidence frame at ${kf.timestamp_sec}s`}
                  onError={e => { e.target.style.display='none' }}
                />
                <div className="evidence-meta">
                  <div>t = {kf.timestamp_sec}s</div>
                  <div>attn = {kf.attention}</div>
                </div>
              </div>
            ))}
          </div>
          {!result.keyframes?.length && (
            <p className="text-muted" style={{ fontSize:'.875rem', marginTop:12 }}>No key frames extracted.</p>
          )}
        </div>
      )}

      {/* ── Report Tab ── */}
      {tab === 'report' && (
        <div className="fade-in">
          <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', marginBottom:16 }}>
            <div style={{ fontWeight:700, fontSize:'1.05rem', display:'flex', alignItems:'center', gap:8 }}>
              <FileText size={18} color="var(--blue)" /> Forensic Analysis Report
            </div>
            <button className="btn btn-primary" onClick={downloadReport}>
              <Download size={14} /> Download PDF
            </button>
          </div>
          {rep ? (
            <div>
              <ReportSection icon="📋" title="Executive Summary"       content={rep.executive_summary} />
              <ReportSection icon="🔎" title="Incident Description"    content={rep.incident_description} />
              <ReportSection icon="⚡" title="Detected Activities"     content={rep.detected_activities} />
              <ReportSection icon="⚠️" title="Threat Assessment"       content={rep.threat_assessment} />
              <ReportSection icon="⏱" title="Chronological Summary"   content={rep.chronological_summary} />
              <ReportSection icon="🔬" title="Investigator Observations" content={rep.investigator_observations} />
              <ReportSection icon="✅" title="Recommended Actions"     content={rep.recommended_actions} />
              <ReportSection icon="🏁" title="Final Conclusion"        content={rep.final_conclusion} />
            </div>
          ) : (
            <div className="card" style={{ textAlign:'center', padding:'40px' }}>
              <p className="text-muted">Report not yet available.</p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
