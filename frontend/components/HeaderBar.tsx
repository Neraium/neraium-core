interface HeaderBarProps {
  frame: any
}

export default function HeaderBar({ frame }: HeaderBarProps) {
  const confidence = frame.confidence ? (frame.confidence * 100).toFixed(0) : '—'
  const phase = frame.phase ? frame.phase.toUpperCase().replace(/_/g, ' ') : 'UNKNOWN'
  const health = frame.system_health ? frame.system_health.toUpperCase() : 'NOMINAL'

  // Format timestamp for display
  const timestamp = frame.timestamp ? new Date(frame.timestamp).toLocaleTimeString() : new Date().toLocaleTimeString()

  return (
    <div className="demo-header">
      <div className="header-brand">
        <span className="brand-wordmark">NERAIUM</span>
        <span className="brand-env">Live System Monitoring</span>
      </div>
      <div className="header-metrics">
        <div className="header-metric">
          <span className="metric-label">System Health</span>
          <span className="metric-value" style={{ color: health === 'NOMINAL' ? '#10b981' : health === 'WATCH' ? '#f59e0b' : '#ef4444' }}>
            {health}
          </span>
        </div>
        <div className="header-metric">
          <span className="metric-label">Confidence</span>
          <span className="metric-value">{confidence}%</span>
        </div>
        <div className="header-metric">
          <span className="metric-label">Last Update</span>
          <span className="metric-value">{timestamp}</span>
        </div>
      </div>
    </div>
  )
}
