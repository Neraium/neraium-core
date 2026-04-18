'use client'

import { DecisionUIState } from '@/lib/decisionToUI'
import StatusHeader from '@/components/StatusHeader'
import ActionPanel from '@/components/ActionPanel'
import DecisionTrace from '@/components/DecisionTrace'
import SystemTimeline from '@/components/SystemTimeline'
import DriftChart from '@/components/DriftChart'
import EnhancedTetrahedronViz from '@/components/EnhancedTetrahedronViz'

interface DecisionUIProps {
  state?: DecisionUIState
  isLoading?: boolean
  error?: string | null
}

export default function DecisionUI({ state, isLoading = false, error = null }: DecisionUIProps) {
  if (error) {
    return (
      <div style={styles.root}>
        <div style={styles.errorContainer}>
          <div style={styles.errorIcon}>⚠</div>
          <div style={styles.errorTitle}>Unable to Load Decision State</div>
          <div style={styles.errorMessage}>{error}</div>
        </div>
      </div>
    )
  }

  if (isLoading || !state) {
    return (
      <div style={styles.root}>
        <div style={styles.loadingContainer}>
          <div style={styles.loadingSpinner} />
          <div style={styles.loadingText}>Loading decision state...</div>
        </div>
      </div>
    )
  }

  return (
    <div style={styles.root}>
      {/* TIER 1: Status - most prominent */}
      <div style={styles.tier1}>
        <StatusHeader status={state.statusHeader} />
      </div>

      {/* TIER 2: Action - immediate next step */}
      <div style={styles.tier2}>
        <ActionPanel action={state.actionPanel} />
      </div>

      {/* TIER 3: Visualizations - context */}
      <div style={styles.tier3}>
        <div style={styles.vizSection}>
          {/* Tetrahedron */}
          <div style={styles.tetrahedronPanel}>
            {state.tetrahedron.trailPoints.length > 0 ? (
              <EnhancedTetrahedronViz tetrahedronState={state.tetrahedron} isInteractive={true} />
            ) : (
              <div style={styles.emptyState}>No historical data</div>
            )}
          </div>

          {/* Timeline and Drift */}
          <div style={styles.rightColumn}>
            <SystemTimeline timeline={state.timeline} />
            {state.driftChart.dataPoints.length > 0 ? (
              <DriftChart chart={state.driftChart} />
            ) : (
              <div style={styles.emptyChartContainer}>
                <div style={styles.emptyState}>No drift data available</div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* TIER 4: Explanation - supporting details */}
      <div style={styles.tier4}>
        <DecisionTrace trace={state.decisionTrace} />
      </div>

      {/* Footer */}
      <div style={styles.footer}>
        <span style={styles.timestamp}>
          {new Date(state.timestamp).toLocaleString(undefined, {
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false,
          })}
        </span>
      </div>
    </div>
  )
}

const styles = {
  root: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '24px',
    padding: '32px',
    backgroundColor: '#000000',
    color: '#ffffff',
    minHeight: '100vh',
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif",
  },

  // Error state
  errorContainer: {
    display: 'flex',
    flexDirection: 'column' as const,
    alignItems: 'center',
    justifyContent: 'center',
    gap: '20px',
    padding: '48px',
    backgroundColor: 'rgba(15, 15, 15, 0.6)',
    borderRadius: '12px',
    border: '1px solid rgba(220, 38, 38, 0.3)',
    minHeight: '300px',
  },

  errorIcon: {
    fontSize: '48px',
    opacity: 0.6,
  },

  errorTitle: {
    fontSize: '16px',
    fontWeight: '600',
    color: '#dc2626',
  },

  errorMessage: {
    fontSize: '13px',
    color: 'rgba(255, 255, 255, 0.6)',
    textAlign: 'center',
  },

  // Loading state
  loadingContainer: {
    display: 'flex',
    flexDirection: 'column' as const,
    alignItems: 'center',
    justifyContent: 'center',
    gap: '20px',
    padding: '48px',
    minHeight: '300px',
  },

  loadingSpinner: {
    width: '32px',
    height: '32px',
    borderRadius: '50%',
    border: '2px solid rgba(255, 255, 255, 0.1)',
    borderTopColor: '#0284c7',
    animation: 'spin 1s linear infinite',
  },

  loadingText: {
    fontSize: '13px',
    color: 'rgba(255, 255, 255, 0.5)',
  },

  // Tier structure
  tier1: {
    order: 1 as any,
  },

  tier2: {
    order: 2 as any,
  },

  tier3: {
    order: 3 as any,
  },

  vizSection: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '24px',
  },

  tetrahedronPanel: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '16px',
    padding: '24px',
    backgroundColor: 'rgba(15, 15, 15, 0.6)',
    borderRadius: '12px',
    border: '1px solid rgba(255, 255, 255, 0.08)',
    backdropFilter: 'blur(4px)',
    minHeight: '520px',
    transition: 'all 0.3s ease',
  },


  emptyState: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    flex: 1,
    minHeight: '400px',
    fontSize: '13px',
    color: 'rgba(255, 255, 255, 0.3)',
    fontStyle: 'italic',
  },

  emptyChartContainer: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '16px',
    padding: '24px',
    backgroundColor: 'rgba(15, 15, 15, 0.6)',
    borderRadius: '12px',
    border: '1px solid rgba(255, 255, 255, 0.08)',
    minHeight: '240px',
  },

  rightColumn: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '24px',
  },

  tier4: {
    order: 4 as any,
    marginTop: '8px',
  },

  footer: {
    display: 'flex',
    justifyContent: 'flex-end',
    paddingTop: '24px',
    borderTop: '1px solid rgba(255, 255, 255, 0.08)',
    marginTop: '16px',
  },

  timestamp: {
    fontSize: '11px',
    color: 'rgba(255, 255, 255, 0.3)',
    fontFamily: 'monospace',
    letterSpacing: '0.05em',
    fontVariantNumeric: 'tabular-nums',
  },
}

if (typeof document !== 'undefined') {
  const style = document.createElement('style')
  style.textContent = `
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
  `
  document.head.appendChild(style)
}
