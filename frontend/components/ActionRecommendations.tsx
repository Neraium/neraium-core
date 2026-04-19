'use client'

import { DecisionUIState, ActionHorizon } from '@/lib/decisionToUI'

interface ActionRecommendationsProps {
  state: DecisionUIState
}

export default function ActionRecommendations({ state }: ActionRecommendationsProps) {
  const horizonLabels: Record<ActionHorizon, string> = {
    [ActionHorizon.NOW]: 'Immediate Action',
    [ActionHorizon.SOON]: 'Near-term Action',
    [ActionHorizon.WATCHLIST]: 'Monitor & Watchlist',
  }

  const horizonColors: Record<ActionHorizon, string> = {
    [ActionHorizon.NOW]: '#ef4444',
    [ActionHorizon.SOON]: '#f97316',
    [ActionHorizon.WATCHLIST]: '#eab308',
  }

  const horizon = state.actionPanel.horizon
  const label = horizonLabels[horizon]
  const color = horizonColors[horizon]

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <div style={styles.title}>What Should I Do?</div>
        <div style={{ ...styles.horizonBadge, borderColor: color }}>
          <div style={{ ...styles.horizonDot, backgroundColor: color }} />
          <span style={styles.horizonLabel}>{label}</span>
        </div>
      </div>

      <div style={styles.content}>
        {/* Primary degradation/observation */}
        <div style={styles.section}>
          <div style={styles.sectionTitle}>Current Situation</div>
          <div style={styles.observation}>
            {state.decisionTrace.primaryFactor}
          </div>
        </div>

        {/* Contributing factors */}
        {state.decisionTrace.secondaryFactors.length > 0 && (
          <div style={styles.section}>
            <div style={styles.sectionTitle}>Supporting Evidence</div>
            <div style={styles.factors}>
              {state.decisionTrace.secondaryFactors.map((factor, idx) => (
                <div key={idx} style={styles.factor}>
                  <span style={styles.factorBullet}>→</span>
                  <span>{factor}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Pattern insight if available */}
        {state.decisionTrace.patternInsight && (
          <div style={styles.section}>
            <div style={styles.sectionTitle}>Pattern Match</div>
            <div style={styles.patternInsight}>
              <div style={styles.patternOutcome}>
                {state.decisionTrace.patternInsight.outcomeType}
              </div>
              <div style={styles.patternSummary}>
                {state.decisionTrace.patternInsight.influenceSummary}
              </div>
            </div>
          </div>
        )}

        {/* Recommended actions */}
        <div style={styles.section}>
          <div style={styles.sectionTitle}>Recommended Actions</div>
          <div style={styles.actionsList}>
            <ActionItem
              title="Primary Action"
              description={state.actionPanel.primaryAction}
              urgency={state.actionPanel.urgencyLevel}
            />
            {horizon === ActionHorizon.NOW && (
              <>
                <ActionItem
                  title="Verify"
                  description="Confirm system state with on-site inspection"
                  urgency={5}
                />
                <ActionItem
                  title="Document"
                  description="Record current system readings and parameters"
                  urgency={5}
                />
              </>
            )}
            {(horizon === ActionHorizon.NOW || horizon === ActionHorizon.SOON) && (
              <ActionItem
                title="Plan Service"
                description="Schedule preventive maintenance within 48 hours"
                urgency={horizon === ActionHorizon.NOW ? 5 : 3}
              />
            )}
            {horizon === ActionHorizon.WATCHLIST && (
              <ActionItem
                title="Monitor Trend"
                description="Continue observing system behavior and trajectory"
                urgency={1}
              />
            )}
          </div>
        </div>
      </div>

      <div style={styles.footer}>
        <div style={styles.footerText}>
          Confidence: {state.statusHeader.confidence} • {state.decisionTrace.confidenceRationale}
        </div>
      </div>
    </div>
  )
}

interface ActionItemProps {
  title: string
  description: string
  urgency: number // 1-5
}

function ActionItem({ title, description, urgency }: ActionItemProps) {
  const urgencyColor = urgency >= 4 ? '#ef4444' : urgency >= 3 ? '#f97316' : urgency >= 2 ? '#eab308' : '#22c55e'
  const urgencyBg = urgency >= 4 ? 'rgba(239, 68, 68, 0.1)' : urgency >= 3 ? 'rgba(249, 115, 22, 0.1)' : urgency >= 2 ? 'rgba(234, 179, 8, 0.1)' : 'rgba(34, 197, 94, 0.1)'

  return (
    <div style={{ ...styles.actionItem, borderLeftColor: urgencyColor, backgroundColor: urgencyBg }}>
      <div style={styles.actionTitle}>{title}</div>
      <div style={styles.actionDescription}>{description}</div>
    </div>
  )
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    display: 'flex',
    flexDirection: 'column',
    gap: '16px',
    padding: '20px',
    backgroundColor: 'rgba(15, 23, 42, 0.3)',
    borderRadius: '12px',
    border: '1px solid rgba(148, 163, 184, 0.08)',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingBottom: '12px',
    borderBottom: '1px solid rgba(148, 163, 184, 0.08)',
  },
  title: {
    fontSize: '12px',
    fontWeight: '600',
    letterSpacing: '0.1em',
    textTransform: 'uppercase',
    color: 'rgba(148, 163, 184, 0.72)',
  },
  horizonBadge: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '6px 12px',
    borderRadius: '6px',
    border: '1px solid',
    backgroundColor: 'rgba(30, 41, 59, 0.6)',
  },
  horizonDot: {
    width: '6px',
    height: '6px',
    borderRadius: '50%',
    flexShrink: 0,
  },
  horizonLabel: {
    fontSize: '11px',
    fontWeight: '500',
    color: '#cbd5e1',
  },
  content: {
    display: 'flex',
    flexDirection: 'column',
    gap: '14px',
  },
  section: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  sectionTitle: {
    fontSize: '10px',
    fontWeight: '600',
    textTransform: 'uppercase',
    color: 'rgba(148, 163, 184, 0.55)',
    letterSpacing: '0.05em',
  },
  observation: {
    fontSize: '13px',
    color: '#e2e8f0',
    lineHeight: 1.5,
    padding: '10px 12px',
    backgroundColor: 'rgba(30, 41, 59, 0.5)',
    borderRadius: '6px',
    borderLeft: '2px solid rgba(96, 165, 250, 0.5)',
  },
  factors: {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
  },
  factor: {
    fontSize: '12px',
    color: '#cbd5e1',
    display: 'flex',
    gap: '8px',
    alignItems: 'flex-start',
    lineHeight: 1.4,
  },
  factorBullet: {
    color: 'rgba(148, 163, 184, 0.5)',
    flexShrink: 0,
  },
  patternInsight: {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
    padding: '10px 12px',
    backgroundColor: 'rgba(30, 41, 59, 0.5)',
    borderRadius: '6px',
  },
  patternOutcome: {
    fontSize: '11px',
    fontWeight: '600',
    color: '#60a5fa',
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
  },
  patternSummary: {
    fontSize: '12px',
    color: '#cbd5e1',
    lineHeight: 1.4,
  },
  actionsList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  actionItem: {
    display: 'flex',
    flexDirection: 'column',
    gap: '4px',
    padding: '10px 12px',
    borderRadius: '6px',
    borderLeft: '3px solid',
  },
  actionTitle: {
    fontSize: '11px',
    fontWeight: '600',
    textTransform: 'uppercase',
    color: '#e2e8f0',
    letterSpacing: '0.05em',
  },
  actionDescription: {
    fontSize: '12px',
    color: '#cbd5e1',
    lineHeight: 1.4,
  },
  footer: {
    paddingTop: '8px',
    borderTop: '1px solid rgba(148, 163, 184, 0.08)',
  },
  footerText: {
    fontSize: '10px',
    color: 'rgba(148, 163, 184, 0.5)',
    fontStyle: 'italic',
  },
}
