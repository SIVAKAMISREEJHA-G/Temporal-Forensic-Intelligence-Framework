import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Clock, FileVideo } from 'lucide-react'

const API = '/api'
const SEVERITY = { Normal:'low','Car Accident':'medium',Abuse:'high',Fighting:'high',Riot:'high',Shooting:'critical',Explosion:'critical' }

export default function HistoryPage() {
  const [videos, setVideos] = useState([])
  const [loading, setLoading] = useState(true)
  const navigate = useNavigate()

  useEffect(() => {
    fetch(`${API}/videos`).then(r=>r.json()).then(data => { setVideos(data); setLoading(false) }).catch(()=>setLoading(false))
  }, [])

  if (loading) return (
    <div style={{ display:'flex', alignItems:'center', justifyContent:'center', height:'60vh', gap:16 }}>
      <div className="spinner" style={{ width:36, height:36 }} />
      <span className="text-muted">Loading history…</span>
    </div>
  )

  return (
    <div className="fade-in">
      <div className="page-header flex items-center justify-between">
        <div>
          <h1>📋 Analysis History</h1>
          <div className="subtitle">{videos.length} video{videos.length!==1?'s':''} analysed</div>
        </div>
        <button className="btn btn-primary" onClick={() => navigate('/upload')}>+ Upload Video</button>
      </div>

      <div className="card">
        {videos.length === 0 ? (
          <div style={{ textAlign:'center', padding:'48px 24px' }}>
            <FileVideo size={48} color="var(--text-muted)" style={{ marginBottom:12 }} />
            <p className="text-muted">No videos analysed yet.</p>
            <button className="btn btn-primary" style={{ marginTop:16 }} onClick={() => navigate('/upload')}>
              Upload First Video
            </button>
          </div>
        ) : (
          <table className="data-table">
            <thead>
              <tr>
                <th>Case ID</th>
                <th>File Name</th>
                <th>Duration</th>
                <th>Class</th>
                <th>Confidence</th>
                <th>Status</th>
                <th>Date</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {videos.map(v => (
                <tr key={v.id}>
                  <td className="text-mono" style={{ fontSize:'.72rem', color:'var(--text-muted)' }}>
                    TFIF-{String(v.id).padStart(6,'0')}
                  </td>
                  <td style={{ maxWidth:200, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }}>
                    {v.orig_name}
                  </td>
                  <td>{v.duration ? `${v.duration.toFixed(1)}s` : '—'}</td>
                  <td>
                    {v.predicted_class ? (
                      <span className={`badge badge-${SEVERITY[v.predicted_class]||'medium'}`}>
                        {v.predicted_class}
                      </span>
                    ) : '—'}
                  </td>
                  <td>{v.confidence ? `${(v.confidence*100).toFixed(1)}%` : '—'}</td>
                  <td>
                    <span className={`badge badge-${v.status||'queued'}`}>
                      {v.status || 'queued'}
                    </span>
                  </td>
                  <td style={{ fontSize:'.75rem', color:'var(--text-muted)', whiteSpace:'nowrap' }}>
                    <Clock size={11} style={{ marginRight:4, verticalAlign:'middle' }} />
                    {v.upload_time ? new Date(v.upload_time+'Z').toLocaleString() : '—'}
                  </td>
                  <td>
                    {v.status === 'done' && (
                      <button className="btn btn-outline" style={{ fontSize:'.72rem', padding:'4px 10px' }}
                        onClick={() => navigate(`/analysis/${v.id}`)}>
                        View →
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
