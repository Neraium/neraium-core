interface HeaderBarProps {
  frame: Record<string, unknown>
}

export default function HeaderBar({ frame }: HeaderBarProps) {
  const confidence = frame.confidence ? ((frame.confidence as number) * 100).toFixed(0) : '—'
  const healthRaw = frame.system_health ? (frame.system_health as string).toUpperCase() : 'NOMINAL'
  const health = healthRaw === 'WARNING' ? 'WATCH' : healthRaw

  // Format timestamp for display
  const timestamp = frame.timestamp ? new Date(frame.timestamp as string).toLocaleTimeString() : new Date().toLocaleTimeString()

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
