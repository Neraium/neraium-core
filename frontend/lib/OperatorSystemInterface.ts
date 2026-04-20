// OperatorSystemInterface.ts
// Complete unified operator decision system
// Handles all decision logic in one place

export interface SystemState {
  drift: number
  coherence: number
  timestamp: string
  subsystems: Subsystem[]
  critical_alerts: string[]
  timeline_states?: TimelineState[]
}

export interface Subsystem {
  subsystem_id: string
  subsystem_name: string
  drift_contribution_pct: number
  confidence: number
  fragility_pct: number
  behavioral_state: string
}

export interface TimelineState {
  drift: number
  timestamp: string
  index: number
}

// ============================================================================
// 1. STAGE CLASSIFICATION
// ============================================================================

export type SystemStage = 'stable' | 'emerging' | 'approaching' | 'critical'

export function classifySystemStage(drift: number): SystemStage {
  if (drift < 0.15) return 'stable'
  if (drift < 0.35) return 'emerging'
  if (drift < 0.6) return 'approaching'
  return 'critical'
}

// ============================================================================
// 2. CONFIDENCE & URGENCY SCORING
// ============================================================================

export function calculateConfidenceScore(drift: number, coherence: number): number {
  const coherenceFactor = coherence * 0.6
  const stabilityFactor = Math.max(0, (1 - drift) * 0.4)
  return Math.round((coherenceFactor + stabilityFactor) * 100)
}

export function classifyConfidence(score: number): 'Low' | 'Medium' | 'High' {
  if (score >= 70) return 'High'
  if (score >= 40) return 'Medium'
  return 'Low'
}

export function calculateUrgencyLevel(drift: number): SystemStage {
  return classifySystemStage(drift)
}

// ============================================================================
// 3. SUBSYSTEM ACTION MAPPING
// ============================================================================

export const subsystemActions: Record<
  string,
  {
    actions: string[]
    inspectionPoints: string[]
    priority: 'primary' | 'secondary'
  }
> = {
  climate: {
    actions: [
      'Verify temperature setpoints and variance limits',
      'Inspect thermal sensing accuracy',
      'Check cooling system capacity and response time',
      'Confirm thermostat calibration',
    ],
    inspectionPoints: [
      'Cooling loop flow rate',
      'Temperature sensor readings',
      'System response lag',
      'Thermal balance across zones',
    ],
    priority: 'primary',
  },
  airflow: {
    actions: [
      'Verify fan speeds and airflow distribution',
      'Check for blockages or restrictions',
      'Inspect damper positions and movement',
      'Confirm air intake and exhaust paths',
    ],
    inspectionPoints: [
      'Fan motor current draw',
      'Airflow pressure differential',
      'Filter condition',
      'Duct seal integrity',
    ],
    priority: 'primary',
  },
  irrigation: {
    actions: [
      'Verify pump pressure and flow rate',
      'Check valve positions and responsiveness',
      'Inspect distribution line integrity',
      'Confirm moisture sensor accuracy',
    ],
    inspectionPoints: [
      'Pump discharge pressure',
      'Valve response time',
      'Drip line patency',
      'Soil moisture sensor reading',
    ],
    priority: 'primary',
  },
  plant: {
    actions: [
      'Assess plant health and vigor',
      'Check for stress indicators',
      'Verify nutrient and water uptake',
      'Inspect for pathogen or pest presence',
    ],
    inspectionPoints: [
      'Leaf color and turgor',
      'Growth rate',
      'Root condition',
      'Transpiration rate',
    ],
    priority: 'secondary',
  },
}

export function getRecommendedActions(subsystemName: string, driftContribution: number): string[] {
  const key = subsystemName.toLowerCase()
  const mapping = subsystemActions[key]

  if (!mapping) return ['Monitor subsystem closely', 'Check sensor readings']

  if (driftContribution > 20) return mapping.actions.slice(0, 2)
  return [mapping.actions[0]]
}

export function getInspectionPoints(subsystemName: string): string[] {
  const key = subsystemName.toLowerCase()
  return subsystemActions[key]?.inspectionPoints || []
}

// ============================================================================
// 4. SITUATION ASSESSMENT
// ============================================================================

export interface SituationAssessment {
  stageLabel: string
  stageColor: string
  description: string
  coherence: number
  drift: number
  primaryDriver: Subsystem | null
  primaryDriverContribution: number
}

export function assessSituation(state: SystemState): SituationAssessment {
  const stage = classifySystemStage(state.drift)
  const primaryDriver = state.subsystems?.reduce((max, sub) =>
    (sub.drift_contribution_pct || 0) > (max.drift_contribution_pct || 0) ? sub : max
  ) || null

  const stageColors = {
    stable: '#10b981',
    emerging: '#f59e0b',
    approaching: '#f97316',
    critical: '#ef4444',
  }

  const descriptions = {
    stable: 'System operating within normal parameters',
    emerging: 'Early drift signals detected',
    approaching: 'Transition becoming likely',
    critical: 'System intervention required',
  }

  const stageLabels = {
    stable: 'STABLE',
    emerging: 'EMERGING',
    approaching: 'APPROACHING TRANSITION',
    critical: 'CRITICAL',
  }

  return {
    stageLabel: stageLabels[stage],
    stageColor: stageColors[stage],
    description: descriptions[stage],
    coherence: state.coherence,
    drift: state.drift,
    primaryDriver,
    primaryDriverContribution: primaryDriver?.drift_contribution_pct || 0,
  }
}

// ============================================================================
// 5. TIME TO IMPACT CALCULATION
// ============================================================================

export interface TimeToImpact {
  display: string
  label: string
  critical: boolean
  cyclesRemaining: number
}

export interface LeadTimeEvent {
  driftOnsetIndex: number
  transitionIndex: number
  leadTimeCycles: number
}

export function detectLeadTime(timeline: TimelineState[]): LeadTimeEvent | null {
  if (!timeline || timeline.length < 3) return null

  // Find drift onset (8% drift + 0.8% acceleration)
  let onsetIdx = -1
  for (let i = 1; i < timeline.length; i++) {
    const acceleration = timeline[i].drift - timeline[i - 1].drift
    if (timeline[i].drift >= 0.08 && acceleration > 0.008) {
      onsetIdx = i
      break
    }
  }

  if (onsetIdx < 0) return null

  // Find first state transition after onset
  const thresholds = [0.2, 0.4, 0.6]
  let lastState = thresholds.filter((t) => timeline[0].drift <= t).length

  for (let i = 1; i < timeline.length; i++) {
    const currentState = thresholds.filter((t) => timeline[i].drift <= t).length
    if (currentState < lastState && i > onsetIdx) {
      return {
        driftOnsetIndex: onsetIdx,
        transitionIndex: i,
        leadTimeCycles: i - onsetIdx,
      }
    }
    lastState = currentState
  }

  return null
}

export function calculateTimeToImpact(
  leadTimeEvent: LeadTimeEvent | null,
  currentIndex: number,
  drift: number
): TimeToImpact {
  if (!leadTimeEvent) {
    if (drift < 0.15) return { display: 'N/A', label: 'Stable operation', critical: false, cyclesRemaining: -1 }
    return { display: '?', label: 'Transition timing uncertain', critical: false, cyclesRemaining: -1 }
  }

  const cyclesRemaining = Math.max(0, leadTimeEvent.transitionIndex - currentIndex)

  if (cyclesRemaining <= 0) {
    return { display: 'Occurred', label: 'Transition passed', critical: true, cyclesRemaining: 0 }
  }
  if (cyclesRemaining <= 5) {
    return {
      display: `${cyclesRemaining}`,
      label: 'cycles remaining',
      critical: true,
      cyclesRemaining,
    }
  }
  return { display: `~${cyclesRemaining}`, label: 'cycles remaining', critical: false, cyclesRemaining }
}

// ============================================================================
// 6. RECOMMENDED ACTIONS
// ============================================================================

export interface ActionGuidance {
  actionLevel: string
  color: string
  nextSteps: string[]
  subsystemActions: string[]
}

export function getActionGuidance(
  stage: SystemStage,
  primaryDriver: Subsystem | null
): ActionGuidance {
  const guidance = {
    stable: {
      actionLevel: 'Monitor',
      color: '#10b981',
      nextSteps: [
        'Continue routine monitoring',
        'Log baseline metrics',
        'Verify all sensors responsive',
      ],
    },
    emerging: {
      actionLevel: 'Watch',
      color: '#f59e0b',
      nextSteps: [
        'Increase observation frequency',
        'Identify primary driver',
        'Prepare mitigation procedures',
      ],
    },
    approaching: {
      actionLevel: 'Prepare',
      color: '#f97316',
      nextSteps: [
        'Alert maintenance standby',
        'Inspect primary driver subsystem',
        'Review corrective procedures',
      ],
    },
    critical: {
      actionLevel: 'Act Now',
      color: '#ef4444',
      nextSteps: [
        'Execute corrective action immediately',
        'Reduce system load if possible',
        'Prepare for manual override',
      ],
    },
  }

  const base = guidance[stage]
  const subsystemActions = primaryDriver
    ? getRecommendedActions(primaryDriver.subsystem_name, primaryDriver.drift_contribution_pct)
    : []

  return {
    ...base,
    subsystemActions,
  }
}

// ============================================================================
// 7. COMPLETE OPERATOR DECISION
// ============================================================================

export interface OperatorDecision {
  situation: SituationAssessment
  timeToImpact: TimeToImpact
  actions: ActionGuidance
  confidence: { score: number; level: 'Low' | 'Medium' | 'High' }
  stressedSubsystemCount: number
  alertCount: number
  systemHealth: number
  shouldTriggerOperatorMoment: boolean
  historicalLeadTime: LeadTimeEvent | null
}

export function generateOperatorDecision(
  state: SystemState,
  currentIndex: number = 0
): OperatorDecision {
  const situation = assessSituation(state)
  const confidence = calculateConfidenceScore(state.drift, state.coherence)
  const stage = classifySystemStage(state.drift)
  const leadTimeEvent = state.timeline_states ? detectLeadTime(state.timeline_states) : null
  const timeToImpact = calculateTimeToImpact(leadTimeEvent, currentIndex, state.drift)
  const actions = getActionGuidance(stage, situation.primaryDriver)

  return {
    situation,
    timeToImpact,
    actions,
    confidence: {
      score: confidence,
      level: classifyConfidence(confidence),
    },
    stressedSubsystemCount: state.subsystems?.filter((s) => (s.drift_contribution_pct || 0) > 10).length || 0,
    alertCount: state.critical_alerts?.length || 0,
    systemHealth: Math.round(state.coherence * 100),
    shouldTriggerOperatorMoment: state.drift >= 0.6,
    historicalLeadTime: leadTimeEvent,
  }
}

// ============================================================================
// 8. DISPLAY UTILITIES
// ============================================================================

export function formatTimeToImpactDisplay(drift: number): string {
  if (drift < 0.35) return '30+ cycles'
  if (drift < 0.6) return '10-20 cycles'
  return '<5 cycles'
}

export function formatWindowStatus(drift: number): string {
  if (drift < 0.35) return 'Adequate window'
  if (drift < 0.6) return 'Narrowing'
  return 'Critical - Act Now'
}

export function formatSituationStatement(primaryDriver: Subsystem | null, drift: number): string {
  if (!primaryDriver) {
    return 'System operating within normal parameters'
  }

  const severity = drift < 0.35 ? 'emerging' : drift < 0.6 ? 'spreading' : 'accelerating'
  return `Structural drift ${severity} in ${primaryDriver.subsystem_name} subsystem, reducing coherence and increasing transition likelihood`
}

// ============================================================================
// 9. USAGE EXAMPLE
// ============================================================================

/*
import { generateOperatorDecision, formatSituationStatement } from './OperatorSystemInterface'

const systemState = {
  drift: 0.45,
  coherence: 0.65,
  timestamp: '2026-04-20T12:00:00Z',
  subsystems: [
    {
      subsystem_id: 'climate',
      subsystem_name: 'Climate',
      drift_contribution_pct: 35,
      confidence: 0.8,
      fragility_pct: 45,
      behavioral_state: 'Coupling',
    },
    // ... more subsystems
  ],
  critical_alerts: [],
  timeline_states: [
    // ... timeline data
  ],
}

const decision = generateOperatorDecision(systemState, 50)

console.log('OPERATOR DECISION:')
console.log('Stage:', decision.situation.stageLabel)
console.log('Situation:', formatSituationStatement(decision.situation.primaryDriver, decision.situation.drift))
console.log('Time to Impact:', decision.timeToImpact.display, decision.timeToImpact.label)
console.log('Confidence:', decision.confidence.level, `(${decision.confidence.score}%)`)
console.log('Action Level:', decision.actions.actionLevel)
console.log('Next Steps:', decision.actions.nextSteps)
console.log('Subsystem Actions:', decision.actions.subsystemActions)
console.log('Alert Operator?', decision.shouldTriggerOperatorMoment)
*/
