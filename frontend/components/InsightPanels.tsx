interface InsightPanelsProps {
  frame: Record<string, unknown>
}

export default function InsightPanels({ frame }: InsightPanelsProps) {
  const getStatusColor = (value: boolean | string | undefined) => {
    if (value === true || value === 'admitted' || value === 'ADMITTED') return '#22C55E'
    if (value === false || value === 'suppressed' || value === 'SUPPRESSED') return '#F97316'
    return '#9CA3AF'
  }

  const health = ((frame.system_health as string) || 'nominal').toLowerCase()
  const healthColor =
    health === 'critical' || health === 'degraded'
      ? '#EF4444'
      : health === 'watch' || health === 'warning'
        ? '#F97316'
        : '#22C55E'
  const transitionType = (frame.transition_type as string) || 'STABLE'

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
          <div className="insight-value" style={{ color: getStatusColor(frame.event_admitted as boolean | string | undefined) }}>
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
          <div className="insight-value">{((frame.persistence_minutes as number) || 0)} min</div>
          <div className="insight-detail">Signal duration</div>
        </div>

        <div className="insight-card">
          <div className="insight-title">Phase</div>
          <div className="insight-value">
            {frame.phase ? (frame.phase as string).toUpperCase().replace(/_/g, ' ') : 'UNKNOWN'}
          </div>
          <div className="insight-detail">Operating regime</div>
        </div>

        <div className="insight-card">
          <div className="insight-title">Timestamp</div>
          <div className="insight-value" style={{ fontSize: '12px' }}>
            {frame.timestamp ? new Date(frame.timestamp as string).toLocaleTimeString() : '—'}
          </div>
          <div className="insight-detail">Frame timestamp</div>
        </div>
      </div>

      <div style={{ marginTop: '28px', paddingTop: '22px', borderTop: '1px solid rgba(255,255,255,0.1)' }}>
        <div className="insight-title" style={{ marginBottom: '16px' }}>
          Metrics Summary
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '14px' }}>
          <div style={{ padding: '12px', background: 'rgba(255,255,255,0.05)', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.08)', transition: 'all 0.2s ease' }} className="insight-card">
            <div style={{ fontSize: '10px', color: '#94a3b8', marginBottom: '6px', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Drift Score</div>
            <div
              style={{
                fontSize: '20px',
                fontWeight: '800',
                background: `linear-gradient(135deg, #3B82F6, #2563EB)`,
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                transition: 'all 0.3s ease',
              }}
            >
              {(((frame.structural_drift_score as number) || 0) * 100).toFixed(1)}%
            </div>
          </div>
          <div style={{ padding: '12px', background: 'rgba(255,255,255,0.05)', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.08)', transition: 'all 0.2s ease' }} className="insight-card">
            <div style={{ fontSize: '10px', color: '#94a3b8', marginBottom: '6px', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Stability</div>
            <div
              style={{
                fontSize: '20px',
                fontWeight: '800',
                background: `linear-gradient(135deg, #22C55E, #16A34A)`,
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                transition: 'all 0.3s ease',
              }}
            >
              {(((frame.relational_stability_score as number) || 0) * 100).toFixed(1)}%
            </div>
          </div>
          <div style={{ padding: '12px', background: 'rgba(255,255,255,0.05)', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.08)', transition: 'all 0.2s ease' }} className="insight-card">
            <div style={{ fontSize: '10px', color: '#94a3b8', marginBottom: '6px', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Coherence</div>
            <div
              style={{
                fontSize: '20px',
                fontWeight: '800',
                background: `linear-gradient(135deg, #06B6D4, #0891B2)`,
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                transition: 'all 0.3s ease',
              }}
            >
              {(((frame.coherence_score as number) || 0) * 100).toFixed(1)}%
            </div>
          </div>
          <div style={{ padding: '12px', background: 'rgba(255,255,255,0.05)', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.08)', transition: 'all 0.2s ease' }} className="insight-card">
            <div style={{ fontSize: '10px', color: '#94a3b8', marginBottom: '6px', fontWeight: '600', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Confidence</div>
            <div
              style={{
                fontSize: '20px',
                fontWeight: '800',
                background: `linear-gradient(135deg, #F97316, #EA580C)`,
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                transition: 'all 0.3s ease',
              }}
            >
              {(((frame.confidence as number) || 0) * 100).toFixed(1)}%
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
