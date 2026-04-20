'use client'

import { useMemo } from 'react'
import {
  subsystemActionMapping,
  urgencyGuidance,
  calculateConfidenceScore,
  classifyConfidence,
  calculateUrgencyLevel,
  getRecommendedActions,
} from '@/lib/actionMapping'

interface OperatorDecisionPanelProps {
  state: any
  leadTimeEvent?: any
}

export function OperatorDecisionPanel({ state, leadTimeEvent }: OperatorDecisionPanelProps) {
  const drift = state.drift || 0
  const coherence = state.coherence || 0

  // Calculate metrics
  const metrics = useMemo(() => {
    const urgencyLevel = calculateUrgencyLevel(drift)
    const confidenceScore = calculateConfidenceScore(drift, coherence)
    const confidenceLevel = classifyConfidence(confidenceScore)
    const guidance = urgencyGuidance[urgencyLevel]

    // Identify primary driver
    const primaryDriver = state.subsystems?.reduce((max: any, sub: any) =>
      (sub.drift_contribution_pct || 0) > (max.drift_contribution_pct || 0) ? sub : max
    ) || null

    // Get recommended actions for primary driver
    const recommendedActions = primaryDriver
      ? getRecommendedActions(primaryDriver.subsystem_name, primaryDriver.drift_contribution_pct)
      : []

    return {
      urgencyLevel,
      confidenceScore,
      confidenceLevel,
      guidance,
      primaryDriver,
      recommendedActions,
    }
  }, [drift, coherence, state.subsystems])

  // Calculate time to impact
  const timeToImpact = useMemo(() => {
    if (!leadTimeEvent) {
      if (drift < 0.15) return { display: 'N/A', label: 'Stable operation', critical: false }
      return { display: '?', label: 'Transition timing uncertain', critical: false }
    }

    const timelineLength = 90
    const cyclesRemaining = Math.max(0, leadTimeEvent.transitionIndex - timelineLength)

    if (cyclesRemaining <= 0) {
      return { display: 'Occurred', label: 'Transition passed', critical: true }
    }
    if (cyclesRemaining <= 5) {
      return { display: `${cyclesRemaining}`, label: 'cycles remaining', critical: true }
    }
    return { display: `~${cyclesRemaining}`, label: 'cycles remaining', critical: false }
  }, [leadTimeEvent, drift])

  return (
    <div className="w-full bg-gradient-to-b from-white/5 to-transparent border border-white/10 rounded-lg overflow-hidden">
      {/* Header with Stage Badge */}
      <div className="p-8 border-b border-white/10">
        <div className="flex items-center justify-between gap-6">
          <h2 className="text-2xl font-light tracking-wider text-white">Operator Decision</h2>
          <div
            className="px-4 py-2 rounded-lg text-sm font-light uppercase tracking-widest"
            style={{
              color: metrics.guidance.color,
              backgroundColor: `${metrics.guidance.color}15`,
              border: `1px solid ${metrics.guidance.color}40`,
            }}
          >
            {metrics.guidance.stageLabel}
          </div>
        </div>
        <div className="mt-3 text-base font-light text-white/70">
          {metrics.guidance.description}
        </div>
      </div>

      {/* Four-Section Decision Panel */}
      <div className="p-8 space-y-8">
        <div className="grid grid-cols-2 gap-8">
          {/* 1. SITUATION */}
          <div
            className="p-6 rounded-lg border"
            style={{
              borderColor: `${metrics.guidance.color}30`,
              backgroundColor: `${metrics.guidance.color}08`,
            }}
          >
            <div
              className="text-xs font-light uppercase tracking-widest mb-4"
              style={{ color: metrics.guidance.color }}
            >
              Situation
            </div>

            {/* System situation statement */}
            <div className="space-y-4">
              {metrics.primaryDriver ? (
                <>
                  <div className="text-sm font-light text-white/90 leading-relaxed">
                    <strong>Structural drift emerging</strong> in{' '}
                    <span style={{ color: metrics.guidance.color }}>
                      {metrics.primaryDriver.subsystem_name} subsystem
                    </span>
                  </div>

                  {/* Key metrics */}
                  <div className="space-y-2 text-xs font-light text-white/60">
                    <div>
                      Coherence:{' '}
                      <span className="text-white/80 font-medium">{Math.round(coherence * 100)}%</span>
                    </div>
                    <div>
                      Drift Intensity:{' '}
                      <span className="text-white/80 font-medium">{Math.round(drift * 100)}%</span>
                    </div>
                    <div>
                      Primary Contributor:{' '}
                      <span className="text-white/80 font-medium">
                        {Math.round(metrics.primaryDriver.drift_contribution_pct)}%
                      </span>
                    </div>
                  </div>
                </>
              ) : (
                <div className="text-sm font-light text-white/70">
                  System operating within normal parameters
                </div>
              )}
            </div>
          </div>

          {/* 2. TIME TO IMPACT */}
          <div
            className="p-6 rounded-lg border"
            style={{
              borderColor: timeToImpact.critical ? '#ef444430' : '#f5a62830',
              backgroundColor: timeToImpact.critical ? '#ef444408' : '#f5a62808',
            }}
          >
            <div
              className="text-xs font-light uppercase tracking-widest mb-4"
              style={{ color: timeToImpact.critical ? '#ef4444' : '#f59e0b' }}
            >
              Time to Impact
            </div>

            <div className="space-y-4">
              <div className="flex items-baseline gap-3">
                <div
                  className="text-4xl font-light"
                  style={{ color: timeToImpact.critical ? '#ef4444' : '#f59e0b' }}
                >
                  {timeToImpact.display}
                </div>
                <div className="text-xs font-light text-white/50">{timeToImpact.label}</div>
              </div>

              {/* Confidence level */}
              <div className="pt-2 border-t border-white/10">
                <div className="text-xs font-light text-white/50 mb-2">
                  Confidence: <span className="text-white/80">{metrics.confidenceLevel}</span>
                </div>
                <div className="text-xs font-light text-white/50">
                  ({metrics.confidenceScore}% based on coherence + stability)
                </div>
              </div>
            </div>
          </div>

          {/* 3. PRIMARY DRIVER */}
          <div
            className="p-6 rounded-lg border"
            style={{
              borderColor: '#ef444430',
              backgroundColor: '#ef444408',
            }}
          >
            <div className="text-xs font-light uppercase tracking-widest mb-4 text-red-400">
              Primary Driver
            </div>

            {metrics.primaryDriver ? (
              <div className="space-y-3">
                <div className="text-sm font-light text-white/90">
                  {metrics.primaryDriver.subsystem_name}
                </div>

                <div className="space-y-2 text-xs font-light text-white/60">
                  <div>
                    Drift Contribution:{' '}
                    <span className="text-white/80 font-medium">
                      {Math.round(metrics.primaryDriver.drift_contribution_pct)}%
                    </span>
                  </div>
                  <div>
                    Fragility:{' '}
                    <span
                      className="font-medium"
                      style={{
                        color:
                          metrics.primaryDriver.fragility_pct > 65
                            ? '#ef4444'
                            : metrics.primaryDriver.fragility_pct > 40
                              ? '#f97316'
                              : '#f59e0b',
                      }}
                    >
                      {Math.round(metrics.primaryDriver.fragility_pct)}%
                    </span>
                  </div>
                </div>

                {/* Inspection points */}
                <div className="pt-3 border-t border-white/10">
                  <div className="text-xs font-light text-white/40 mb-2">Inspect:</div>
                  <div className="space-y-1 text-xs font-light text-white/60">
                    {subsystemActionMapping[metrics.primaryDriver.subsystem_name.toLowerCase()]
                      ?.inspectionPoints.slice(0, 2)
                      .map((point, idx) => (
                        <div key={idx}>• {point}</div>
                      ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-xs font-light text-white/50">No primary driver detected</div>
            )}
          </div>

          {/* 4. RECOMMENDED ACTION */}
          <div
            className="p-6 rounded-lg border"
            style={{
              borderColor: `${metrics.guidance.color}30`,
              backgroundColor: `${metrics.guidance.color}08`,
            }}
          >
            <div
              className="text-xs font-light uppercase tracking-widest mb-4"
              style={{ color: metrics.guidance.color }}
            >
              Action Level: {metrics.guidance.actionLevel}
            </div>

            <div className="space-y-4">
              {/* Next steps */}
              <div className="space-y-2">
                {metrics.guidance.nextSteps.map((step, idx) => (
                  <div key={idx} className="text-sm font-light text-white/80 flex gap-3">
                    <span className="text-white/40">→</span>
                    <span>{step}</span>
                  </div>
                ))}
              </div>

              {/* Recommended actions for primary driver */}
              {metrics.recommendedActions.length > 0 && (
                <div className="pt-3 border-t border-white/10">
                  <div className="text-xs font-light text-white/50 mb-2">
                    {metrics.primaryDriver?.subsystem_name} specific:
                  </div>
                  <div className="space-y-1">
                    {metrics.recommendedActions.map((action, idx) => (
                      <div key={idx} className="text-xs font-light text-white/70 flex gap-2">
                        <span className="text-white/30">◦</span>
                        <span>{action}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Quick Reference Footer */}
        <div className="border-t border-white/10 pt-6">
          <div className="grid grid-cols-4 gap-4 text-center text-xs font-light">
            <div>
              <div className="text-white/40 mb-2">Stressed Subsystems</div>
              <div className="text-2xl font-light text-white/80">
                {state.subsystems?.filter((s: any) => (s.drift_contribution_pct || 0) > 10).length || 0}
              </div>
            </div>
            <div>
              <div className="text-white/40 mb-2">Alerts</div>
              <div className="text-2xl font-light text-white/80">
                {state.critical_alerts?.length || 0}
              </div>
            </div>
            <div>
              <div className="text-white/40 mb-2">Recovery Likelihood</div>
              <div className="text-2xl font-light text-white/80">
                {coherence > 0.7 ? 'High' : coherence > 0.4 ? 'Medium' : 'Low'}
              </div>
            </div>
            <div>
              <div className="text-white/40 mb-2">System Health</div>
              <div
                className="text-2xl font-light"
                style={{
                  color:
                    coherence > 0.7
                      ? '#10b981'
                      : coherence > 0.4
                        ? '#f59e0b'
                        : '#ef4444',
                }}
              >
                {Math.round(coherence * 100)}%
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
