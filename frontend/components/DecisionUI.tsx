'use client'

import { DecisionUIState } from '@/lib/decisionToUI'
import StatusHeader from '@/components/StatusHeader'
import ActionPanel from '@/components/ActionPanel'
import DecisionTrace from '@/components/DecisionTrace'
import SystemTimeline from '@/components/SystemTimeline'
import DriftChart from '@/components/DriftChart'
import EnhancedTetrahedronViz from '@/components/EnhancedTetrahedronViz'

interface DecisionUIProps {
  state: DecisionUIState
}

export default function DecisionUI({ state }: DecisionUIProps) {
  return (
    <div style={styles.root}>
      {/* TIER 1: Status (most prominent) */}
      <div style={styles.tier1}>
        <StatusHeader status={state.statusHeader} />
      </div>

      {/* TIER 2: Action (immediate next step) */}
      <div style={styles.tier2}>
        <ActionPanel action={state.actionPanel} />
      </div>

      {/* TIER 3: Visualizations (context for understanding) */}
      <div style={styles.tier3}>
        <div style={styles.vizSection}>
          {/* Tetrahedron - spatial representation */}
          <div style={styles.tetrahedronPanel}>
            <EnhancedTetrahedronViz tetrahedronState={state.tetrahedron} isInteractive={true} />
          </div>

          {/* Timeline and Drift - temporal context */}
          <div style={styles.rightColumn}>
            <SystemTimeline timeline={state.timeline} />
            <DriftChart chart={state.driftChart} />
          </div>
        </div>
      </div>

      {/* TIER 4: Explanation (supporting details) */}
      <div style={styles.tier4}>
        <DecisionTrace trace={state.decisionTrace} />
      </div>

      {/* Timestamp footer */}
      <div style={styles.footer}>
        <span style={styles.timestamp}>Last updated: {state.timestamp}</span>
      </div>
    </div>
  )
}

const styles = {
  root: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '20px',
    padding: '24px',
    backgroundColor: '#000000',
    color: '#ffffff',
    minHeight: '100vh',
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif",
  },

  // TIER 1: Status - most prominent, largest
  tier1: {
    order: 1,
  },

  // TIER 2: Action - clear and immediate
  tier2: {
    order: 2,
  },

  // TIER 3: Visualizations - support understanding
  tier3: {
    order: 3,
  },

  vizSection: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: '20px',
  },

  tetrahedronPanel: {
    backgroundColor: 'rgba(30, 30, 30, 0.8)',
    borderRadius: '8px',
    border: '1px solid rgba(255, 255, 255, 0.1)',
    overflow: 'hidden',
    minHeight: '500px',
  },

  rightColumn: {
    display: 'flex',
    flexDirection: 'column' as const,
    gap: '20px',
  },

  // TIER 4: Explanation - supporting details
  tier4: {
    order: 4,
    marginTop: '12px',
  },

  // Footer
  footer: {
    display: 'flex',
    justifyContent: 'flex-end',
    paddingTop: '20px',
    borderTop: '1px solid rgba(255, 255, 255, 0.1)',
    marginTop: '20px',
  },

  timestamp: {
    fontSize: '11px',
    color: 'rgba(255, 255, 255, 0.3)',
    fontFamily: 'monospace',
    letterSpacing: '0.05em',
  },
}
