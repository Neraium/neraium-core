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
    borderBottom: '1px solid rgba(148, 163, 184, 0.1)',
  },
  title: {
    fontSize: '12px',
    fontWeight: '600',
    letterSpacing: '0.1em',
    textTransform: 'uppercase',
    color: 'rgba(148, 163, 184, 0.72)',
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
    padding: '12px 14px',
    backgroundColor: 'rgba(15, 23, 42, 0.4)',
    borderRadius: '12px',
    border: '1px solid rgba(148, 163, 184, 0.08)',
    transition: 'all 0.3s ease',
  },
  metricDot: {
    width: '8px',
    height: '8px',
    borderRadius: '50%',
    flexShrink: 0,
  },
  metricContent: {
    display: 'flex',
    flexDirection: 'column',
    gap: '2px',
  },
  metricValue: {
    fontSize: '18px',
    fontWeight: '700',
    color: '#f1f5f9',
    lineHeight: 1,
  },
  metricLabel: {
    fontSize: '10px',
    color: 'rgba(148, 163, 184, 0.6)',
    fontWeight: '500',
    lineHeight: 1,
  },
}
