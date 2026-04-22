'use client'

interface SystemHealthMetrics {
  healthy: number
  drifting: number
  unstable: number
  earlyWarnings: number
  enteringInstability: number
}

interface SystemOverviewProps {
  metrics?: SystemHealthMetrics
}

export default function SystemOverview({ metrics }: SystemOverviewProps) {
  const defaultMetrics: SystemHealthMetrics = {
    healthy: 12,
    drifting: 3,
    unstable: 1,
    earlyWarnings: 5,
    enteringInstability: 2,
  }

  const data = metrics || defaultMetrics

  return (
    <div style={styles.container}>
      <div style={styles.title}>System Overview</div>

      <div style={styles.metricsGrid}>
        {/* Healthy systems */}
        <div style={styles.metricCard}>
          <div style={{ ...styles.metricDot, backgroundColor: '#22c55e' }} />
          <div style={styles.metricContent}>
            <div style={styles.metricValue}>{data.healthy}</div>
            <div style={styles.metricLabel}>Systems healthy</div>
          </div>
        </div>

        {/* Drifting systems */}
        <div style={styles.metricCard}>
          <div style={{ ...styles.metricDot, backgroundColor: '#eab308' }} />
          <div style={styles.metricContent}>
            <div style={styles.metricValue}>{data.drifting}</div>
            <div style={styles.metricLabel}>Systems drifting</div>
          </div>
        </div>

        {/* Unstable systems */}
        <div style={styles.metricCard}>
          <div style={{ ...styles.metricDot, backgroundColor: '#ef4444' }} />
          <div style={styles.metricContent}>
            <div style={styles.metricValue}>{data.unstable}</div>
            <div style={styles.metricLabel}>Systems unstable</div>
          </div>
        </div>

        {/* Early warnings */}
        <div style={styles.metricCard}>
          <div style={{ ...styles.metricDot, backgroundColor: '#f97316' }} />
          <div style={styles.metricContent}>
            <div style={styles.metricValue}>{data.earlyWarnings}</div>
            <div style={styles.metricLabel}>Active early warnings</div>
          </div>
        </div>

        {/* Entering instability */}
        <div style={styles.metricCard}>
          <div style={{ ...styles.metricDot, backgroundColor: '#ec4899' }} />
          <div style={styles.metricContent}>
            <div style={styles.metricValue}>{data.enteringInstability}</div>
            <div style={styles.metricLabel}>Entering instability</div>
          </div>
        </div>
      </div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
    paddingBottom: '20px',
    borderBottom: '2px solid rgba(0, 102, 255, 0.2)',
  },
  title: {
    fontSize: '11px',
    fontWeight: '700',
    letterSpacing: '0.15em',
    textTransform: 'uppercase',
    color: '#0099ff',
    textShadow: '0 0 8px rgba(0, 102, 255, 0.6)',
  },
  metricsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
    gap: '12px',
  },
  metricCard: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    padding: '14px 16px',
    backgroundColor: 'rgba(0, 20, 40, 0.3)',
    borderRadius: '0',
    border: '1px solid rgba(0, 102, 255, 0.15)',
    backdropFilter: 'blur(8px)',
    transition: 'all 0.3s ease',
    boxShadow: '0 0 15px rgba(0, 102, 255, 0.1), inset 0 0 10px rgba(0, 102, 255, 0.05)',
  },
  metricDot: {
    width: '10px',
    height: '10px',
    borderRadius: '50%',
    flexShrink: 0,
    boxShadow: '0 0 12px currentColor',
  },
  metricContent: {
    display: 'flex',
    flexDirection: 'column',
    gap: '2px',
  },
  metricValue: {
    fontSize: '20px',
    fontWeight: '900',
    color: '#0099ff',
    lineHeight: 1,
    fontFamily: "'Space Mono', monospace",
    textShadow: '0 0 10px rgba(0, 102, 255, 0.8)',
  },
  metricLabel: {
    fontSize: '9px',
    color: '#0088ff',
    fontWeight: '600',
    lineHeight: 1,
    letterSpacing: '0.08em',
    textTransform: 'uppercase',
  },
}

// Add neon effects and holographic styling
if (typeof document !== 'undefined') {
  const style = document.createElement('style')
  style.textContent = `
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');
  `
  document.head.appendChild(style)
}
