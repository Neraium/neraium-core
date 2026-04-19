'use client'

interface OperatorDecisionPanelProps {
  state: any
  leadTimeEvent?: any
}

export function OperatorDecisionPanel({ state, leadTimeEvent }: OperatorDecisionPanelProps) {
  const drift = state.drift || 0
  const coherence = state.coherence || 0

  // Classify system stage
  const getSystemStage = () => {
    if (drift < 0.15) return { stage: 'STABLE', color: '#10b981', description: 'Normal operation' }
    if (drift < 0.35) return { stage: 'EMERGING', color: '#f59e0b', description: 'Early drift signals' }
    if (drift < 0.6) return { stage: 'APPROACHING', color: '#f97316', description: 'Transition imminent' }
    return { stage: 'CRITICAL', color: '#ef4444', description: 'System stressed' }
  }

  const systemStage = getSystemStage()

  // Identify primary driver (subsystem with highest drift contribution)
  const getPrimaryDriver = () => {
    if (!state.subsystems || state.subsystems.length === 0) return null
    return state.subsystems.reduce((max: any, sub: any) =>
      (sub.drift_contribution_pct || 0) > (max.drift_contribution_pct || 0) ? sub : max
    )
  }

  const primaryDriver = getPrimaryDriver()

  // Calculate time to impact
  const getTimeToImpact = () => {
    if (!leadTimeEvent) {
      if (drift < 0.15) return { cycles: 'N/A', label: 'Not in drift window' }
      return { cycles: '?', label: 'Transition timing uncertain' }
    }

    const timelineLength = 90 // cycles (90 seconds)
    const cyclesRemaining = Math.max(0, leadTimeEvent.transitionIndex - timelineLength)

    if (cyclesRemaining <= 0) {
      return { cycles: 'Occurred', label: 'Transition already passed' }
    }
    if (cyclesRemaining <= 5) {
      return { cycles: cyclesRemaining, label: 'cycles', critical: true }
    }
    return { cycles: cyclesRemaining, label: 'cycles' }
  }

  const timeToImpact = getTimeToImpact()

  // Determine recommended action
  const getRecommendedAction = () => {
    if (drift < 0.15) {
      return {
        action: 'Continue Monitoring',
        priority: 'ROUTINE',
        color: '#10b981',
        description: 'No intervention needed',
      }
    }
    if (drift < 0.35) {
      return {
        action: 'Increase Observation',
        priority: 'CAUTION',
        color: '#f59e0b',
        description: 'Watch primary driver closely',
      }
    }
    if (drift < 0.6) {
      return {
        action: 'Prepare Intervention',
        priority: 'WARNING',
        color: '#f97316',
        description: 'Be ready to act',
      }
    }
    return {
      action: 'Intervene Now',
      priority: 'CRITICAL',
      color: '#ef4444',
      description: 'Immediate action required',
    }
  }

  const recommendedAction = getRecommendedAction()

  return (
    <div className="w-full bg-gradient-to-b from-white/5 to-transparent border border-white/10 rounded-lg overflow-hidden">
      <div className="p-8 space-y-8">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-white/10 pb-6">
          <h2 className="text-xl font-light tracking-wider text-white">Operator Decision Panel</h2>
          <div
            className="px-3 py-1 rounded text-xs font-light uppercase tracking-widest"
            style={{ color: systemStage.color, backgroundColor: `${systemStage.color}15` }}
          >
            {systemStage.stage}
          </div>
        </div>

        {/* 4-Section Layout */}
        <div className="grid grid-cols-2 gap-8">
          {/* 1. SITUATION */}
          <div className="space-y-3">
            <div className="text-xs font-light text-white/40 uppercase tracking-widest">Situation</div>
            <div
              className="text-sm font-light leading-relaxed"
              style={{ color: systemStage.color }}
            >
              {systemStage.description}
            </div>
            <div className="text-xs font-light text-white/50 mt-2">
              Coherence: <span className="text-white/70 font-medium">{Math.round(coherence * 100)}%</span>
            </div>
            <div className="text-xs font-light text-white/50">
              Drift: <span className="text-white/70 font-medium">{Math.round(drift * 100)}%</span>
            </div>
          </div>

          {/* 2. TIME TO IMPACT */}
          <div className="space-y-3">
            <div className="text-xs font-light text-white/40 uppercase tracking-widest">Time to Impact</div>
            <div className="flex items-baseline gap-2">
              <div
                className="text-3xl font-light"
                style={{ color: timeToImpact.critical ? '#ef4444' : '#f59e0b' }}
              >
                {timeToImpact.cycles}
              </div>
              <div className="text-xs font-light text-white/50">{timeToImpact.label}</div>
            </div>
            <div className="text-xs font-light text-white/40 mt-2">
              {timeToImpact.critical ? (
                <span className="text-red-400">⚠ Window closing</span>
              ) : timeToImpact.cycles === 'N/A' ? (
                <span className="text-white/60">System stable</span>
              ) : (
                <span className="text-white/60">Lead window active</span>
              )}
            </div>
          </div>

          {/* 3. PRIMARY DRIVER */}
          <div className="space-y-3">
            <div className="text-xs font-light text-white/40 uppercase tracking-widest">Primary Driver</div>
            {primaryDriver ? (
              <>
                <div className="text-sm font-light text-white/90">{primaryDriver.subsystem_name}</div>
                <div className="text-xs font-light text-white/50 space-y-1">
                  <div>
                    Drift Contribution: <span className="text-white/70">{Math.round(primaryDriver.drift_contribution_pct)}%</span>
                  </div>
                  <div>
                    Fragility: <span className="text-white/70">{Math.round(primaryDriver.fragility_pct)}%</span>
                  </div>
                </div>
              </>
            ) : (
              <div className="text-xs font-light text-white/50">No major drivers detected</div>
            )}
          </div>

          {/* 4. RECOMMENDED ACTION */}
          <div className="space-y-3">
            <div className="text-xs font-light text-white/40 uppercase tracking-widest">Recommended Action</div>
            <div
              className="px-3 py-2 rounded border"
              style={{ borderColor: `${recommendedAction.color}40`, backgroundColor: `${recommendedAction.color}08` }}
            >
              <div
                className="text-sm font-light"
                style={{ color: recommendedAction.color }}
              >
                {recommendedAction.action}
              </div>
              <div className="text-xs font-light text-white/50 mt-1">
                {recommendedAction.description}
              </div>
            </div>
          </div>
        </div>

        {/* Quick Facts Footer */}
        <div className="border-t border-white/5 pt-6">
          <div className="grid grid-cols-3 gap-4 text-xs font-light text-white/40">
            <div>
              <div className="text-white/60 mb-1">Subsystems Stressed</div>
              <div className="text-lg font-light text-white/80">
                {state.subsystems?.filter((s: any) => (s.drift_contribution_pct || 0) > 10).length || 0}
              </div>
            </div>
            <div>
              <div className="text-white/60 mb-1">Critical Alerts</div>
              <div className="text-lg font-light text-white/80">
                {state.critical_alerts?.length || 0}
              </div>
            </div>
            <div>
              <div className="text-white/60 mb-1">Recovery Likelihood</div>
              <div className="text-lg font-light text-white/80">
                {coherence > 0.7 ? 'High' : coherence > 0.4 ? 'Medium' : 'Low'}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
