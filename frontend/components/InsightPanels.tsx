interface InsightPanelsProps {
  frame: any
}

export default function InsightPanels({ frame }: InsightPanelsProps) {
  const getStatusColor = (value: boolean | string | undefined) => {
    if (value === true || value === 'admitted' || value === 'ADMITTED') return '#22C55E'
    if (value === false || value === 'suppressed' || value === 'SUPPRESSED') return '#F97316'
    return '#9CA3AF'
  }

  const health = frame.system_health || 'nominal'
  const healthColor = health === 'degraded' ? '#EF4444' : health === 'watch' ? '#F97316' : '#22C55E'
  const transitionType = frame.transition_type || 'STABLE'

  return (
    <div className="panel">
      <div className="panel-head">
        <span className="eyebrow">System Insights</span>
        <span className="panel-subtitle">Real-time operational metrics and state assessment</span>
      </div>
      <div className="insights-grid">
        <div className="insight-card">
          <div className="insight-title">System Health</div>
          <div className="insight-value" style={{ color: healthColor }}>
            {health.toUpperCase()}
          </div>
          <div className="insight-detail">Current operational status</div>
        </div>

        <div className="insight-card">
          <div className="insight-title">Event Status</div>
          <div className="insight-value" style={{ color: getStatusColor(frame.event_admitted) }}>
            {frame.event_admitted ? 'ADMITTED' : 'SUPPRESSED'}
          </div>
          <div className="insight-detail">Gate decision on signal</div>
        </div>

        <div className="insight-card">
          <div className="insight-title">Transition Type</div>
          <div className="insight-value">
            {transitionType.charAt(0) + transitionType.slice(1).toLowerCase()}
          </div>
          <div className="insight-detail">Current phase classification</div>
        </div>

        <div className="insight-card">
          <div className="insight-title">Persistence</div>
          <div className="insight-value">{frame.persistence_minutes || 0} min</div>
          <div className="insight-detail">Signal duration</div>
        </div>

        <div className="insight-card">
          <div className="insight-title">Phase</div>
          <div className="insight-value">
            {frame.phase ? frame.phase.toUpperCase().replace(/_/g, ' ') : 'UNKNOWN'}
          </div>
          <div className="insight-detail">Operating regime</div>
        </div>

        <div className="insight-card">
          <div className="insight-title">Timestamp</div>
          <div className="insight-value" style={{ fontSize: '12px' }}>
            {frame.timestamp ? new Date(frame.timestamp).toLocaleTimeString() : '—'}
          </div>
          <div className="insight-detail">Frame timestamp</div>
        </div>
      </div>

      <div style={{ marginTop: '24px', paddingTop: '20px', borderTop: '1px solid rgba(255,255,255,0.08)' }}>
        <div className="insight-title" style={{ marginBottom: '12px' }}>
          Metrics Summary
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '12px' }}>
          <div style={{ padding: '8px', background: 'rgba(255,255,255,0.03)', borderRadius: '4px' }}>
            <div style={{ fontSize: '10px', color: '#9CA3AF', marginBottom: '4px' }}>Drift Score</div>
            <div
              style={{
                fontSize: '18px',
                fontWeight: '700',
                background: `linear-gradient(90deg, #3B82F6, transparent)`,
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              {(frame.structural_drift_score * 100).toFixed(0)}%
            </div>
          </div>
          <div style={{ padding: '8px', background: 'rgba(255,255,255,0.03)', borderRadius: '4px' }}>
            <div style={{ fontSize: '10px', color: '#9CA3AF', marginBottom: '4px' }}>Stability</div>
            <div
              style={{
                fontSize: '18px',
                fontWeight: '700',
                background: `linear-gradient(90deg, #22C55E, transparent)`,
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              {(frame.relational_stability_score * 100).toFixed(0)}%
            </div>
          </div>
          <div style={{ padding: '8px', background: 'rgba(255,255,255,0.03)', borderRadius: '4px' }}>
            <div style={{ fontSize: '10px', color: '#9CA3AF', marginBottom: '4px' }}>Coherence</div>
            <div
              style={{
                fontSize: '18px',
                fontWeight: '700',
                background: `linear-gradient(90deg, #06B6D4, transparent)`,
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              {(frame.coherence_score * 100).toFixed(0)}%
            </div>
          </div>
          <div style={{ padding: '8px', background: 'rgba(255,255,255,0.03)', borderRadius: '4px' }}>
            <div style={{ fontSize: '10px', color: '#9CA3AF', marginBottom: '4px' }}>Confidence</div>
            <div
              style={{
                fontSize: '18px',
                fontWeight: '700',
                background: `linear-gradient(90deg, #F97316, transparent)`,
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
              }}
            >
              {(frame.confidence * 100).toFixed(0)}%
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
