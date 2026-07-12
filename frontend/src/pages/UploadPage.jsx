import { useState, useCallback, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { useDropzone } from 'react-dropzone'
import { Upload, FileVideo, CheckCircle, XCircle, Loader2 } from 'lucide-react'

const API = '/api'

const STAGES = ['upload','preprocessing','classifying','temporal_reasoning','generating_report','complete']
const STAGE_LABEL = {
  upload:'Upload','preprocessing':'Preprocessing','classifying':'Classifying',
  'temporal_reasoning':'Timeline Analysis','generating_report':'Generating Report','complete':'Complete'
}

function PipelineSteps({ current }) {
  const stageIdx = idx => STAGES.indexOf(idx)
  const curIdx   = stageIdx(current)
  return (
    <div className="pipeline">
      {STAGES.slice(1).map((s, i) => {
        const idx = i + 1
        const done   = idx < curIdx
        const active = idx === curIdx
        return (
          <div key={s} style={{ display:'flex', alignItems:'center', gap:8 }}>
            {i > 0 && <div className="pipeline-arrow">→</div>}
            <div className={`pipeline-step ${done ? 'done' : active ? 'active' : 'pending'}`}>
              {done ? <CheckCircle size={12} /> : active ? <Loader2 size={12} style={{ animation:'spin .7s linear infinite' }} /> : null}
              {STAGE_LABEL[s]}
            </div>
          </div>
        )
      })}
    </div>
  )
}

export default function UploadPage() {
  const navigate   = useNavigate()
  const [file, setFile]       = useState(null)
  const [state, setState]     = useState('idle') // idle | uploading | processing | done | error
  const [videoId, setVideoId] = useState(null)
  const [jobStatus, setJobStatus] = useState(null)
  const [progress, setProgress]   = useState(0)
  const [errorMsg, setErrorMsg]   = useState('')
  const pollRef = useRef(null)

  const onDrop = useCallback(accepted => {
    if (accepted.length > 0) setFile(accepted[0])
  }, [])
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: { 'video/*': ['.mp4','.avi','.mkv','.mov'] }, multiple: false,
  })

  const upload = async () => {
    if (!file) return
    setState('uploading'); setProgress(10); setErrorMsg('')
    try {
      const fd = new FormData()
      fd.append('file', file)
      const res  = await fetch(`${API}/videos/upload`, { method:'POST', body:fd })
      const data = await res.json()
      if (!res.ok) throw new Error(data.detail || 'Upload failed')
      setVideoId(data.video_id)
      setState('processing'); setProgress(20)
      startPolling(data.video_id)
    } catch(e) {
      setErrorMsg(e.message); setState('error')
    }
  }

  const startPolling = id => {
    pollRef.current = setInterval(async () => {
      try {
        const r    = await fetch(`${API}/videos/${id}/status`)
        const data = await r.json()
        setJobStatus(data)
        const p = { queued:15, processing:40, classifying:55, temporal_reasoning:70, generating_report:85, complete:100 }
        setProgress(p[data.stage] || p[data.status] || 50)
        if (data.status === 'done')   { clearInterval(pollRef.current); setState('done') }
        if (data.status === 'failed') { clearInterval(pollRef.current); setState('error'); setErrorMsg(data.message || 'Processing failed') }
      } catch {}
    }, 2000)
  }

  useEffect(() => () => clearInterval(pollRef.current), [])

  return (
    <div className="fade-in" style={{ maxWidth: 760 }}>
      <div className="page-header">
        <h1>📤 Upload Surveillance Video</h1>
        <div className="subtitle">AI analysis begins immediately after upload</div>
      </div>

      {state === 'idle' && (
        <>
          <div {...getRootProps()} className={`upload-zone${isDragActive ? ' drag-over' : ''}`}>
            <input {...getInputProps()} />
            <div className="upload-icon">📹</div>
            {file ? (
              <>
                <h3 style={{ color:'var(--green)' }}>{file.name}</h3>
                <p>{(file.size / 1e6).toFixed(2)} MB · Ready to upload</p>
              </>
            ) : (
              <>
                <h3>Drop video here or click to browse</h3>
                <p>MP4, AVI, MKV, MOV · Any length supported</p>
              </>
            )}
          </div>
          {file && (
            <div style={{ display:'flex', gap:12, marginTop:16 }}>
              <button className="btn btn-primary" onClick={upload}>
                <Upload size={15} /> Analyse Video
              </button>
              <button className="btn btn-outline" onClick={() => setFile(null)}>Clear</button>
            </div>
          )}
        </>
      )}

      {state === 'uploading' && (
        <div className="card">
          <div style={{ display:'flex', alignItems:'center', gap:12, marginBottom:16 }}>
            <Loader2 size={20} style={{ animation:'spin .7s linear infinite', color:'var(--blue)' }} />
            <span>Uploading {file?.name}…</span>
          </div>
          <div className="progress-bar-wrap"><div className="progress-bar" style={{ width:`${progress}%` }} /></div>
        </div>
      )}

      {(state === 'processing') && (
        <div className="card">
          <div className="flex items-center justify-between mb-16">
            <div style={{ display:'flex', alignItems:'center', gap:10 }}>
              <div className="spinner" />
              <div>
                <div style={{ fontWeight:600 }}>{file?.name}</div>
                <div style={{ fontSize:'.8rem', color:'var(--text-secondary)' }}>Case ID: TFIF-{String(videoId).padStart(6,'0')}</div>
              </div>
            </div>
            <span className={`badge badge-${jobStatus?.status || 'processing'}`}>{jobStatus?.stage || 'processing'}</span>
          </div>
          <div className="progress-bar-wrap" style={{ marginBottom:16 }}>
            <div className="progress-bar" style={{ width:`${progress}%` }} />
          </div>
          <PipelineSteps current={jobStatus?.stage || 'preprocessing'} />
        </div>
      )}

      {state === 'done' && (
        <div className="card" style={{ textAlign:'center', padding:'40px 24px' }}>
          <CheckCircle size={48} color="var(--green)" style={{ marginBottom:16 }} />
          <h2 style={{ marginBottom:8 }}>Analysis Complete!</h2>
          <p className="text-muted" style={{ marginBottom:20 }}>Case TFIF-{String(videoId).padStart(6,'0')} · Forensic report generated</p>
          <div style={{ display:'flex', justifyContent:'center', gap:12 }}>
            <button className="btn btn-primary" onClick={() => navigate(`/analysis/${videoId}`)}>
              View Analysis →
            </button>
            <button className="btn btn-outline" onClick={() => { setFile(null); setState('idle'); setVideoId(null); setJobStatus(null); setProgress(0) }}>
              Upload Another
            </button>
          </div>
        </div>
      )}

      {state === 'error' && (
        <div className="card" style={{ borderColor:'var(--red)' }}>
          <div style={{ display:'flex', gap:12, alignItems:'center', marginBottom:8 }}>
            <XCircle size={24} color="var(--red)" />
            <span style={{ fontWeight:600, color:'var(--red)' }}>Processing Failed</span>
          </div>
          <p style={{ fontSize:'.875rem', color:'var(--text-secondary)', marginBottom:16 }}>{errorMsg}</p>
          <button className="btn btn-outline" onClick={() => { setFile(null); setState('idle'); setErrorMsg('') }}>Try Again</button>
        </div>
      )}

      {/* Info */}
      <div className="card mt-24" style={{ background:'rgba(31,111,235,.05)', borderColor:'rgba(31,111,235,.2)' }}>
        <div style={{ fontWeight:600, marginBottom:8, color:'var(--blue)' }}>ℹ️ Processing Pipeline</div>
        <p style={{ fontSize:'.83rem', color:'var(--text-secondary)', lineHeight:1.7 }}>
          After upload, TFIF runs: <strong>frame extraction → MobileNetV3 embedding → BiLSTM classification → temporal segmentation → LLM forensic report</strong>. Processing takes 10–60s depending on video length.
        </p>
      </div>
    </div>
  )
}
