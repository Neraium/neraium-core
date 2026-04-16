interface HeaderBarProps {
  frame: any
}

export default function HeaderBar({ frame }: HeaderBarProps) {
  const confidence = frame.confidence ? (frame.confidence * 100).toFixed(0) : '—'
  const phase = frame.phase ? frame.phase.toUpperCase().replace(/_/g, ' ') : 'UNKNOWN'
  const index = frame.index || 0
  const total = frame.total || 0

  return (
    <div className="demo-header">
      <div className="header-brand">
        <span className="brand-wordmark">NERAIUM</span>
        <span className="brand-env">System Intelligence</span>
      </div>
      <div className="header-metrics">
        <div className="header-metric">
          <span className="metric-label">Confidence</span>
          <span className="metric-value">{confidence}%</span>
        </div>
        <div className="header-metric">
          <span className="metric-label">Phase</span>
          <span className="metric-value">{phase}</span>
        </div>
        <div className="header-metric">
          <span className="metric-label">Frame</span>
          <span className="metric-value">{index} / {total}</span>
        </div>
      </div>
    </div>
  )
}
