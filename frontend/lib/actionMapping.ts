// Subsystem-to-Action Mapping
// Maps subsystem issues to concrete operator actions

export const subsystemActionMapping: Record<string, {
  actions: string[]
  inspectionPoints: string[]
  priority: 'primary' | 'secondary'
}> = {
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

// Urgency-based action prioritization
export const urgencyGuidance: Record<string, {
  stageLabel: string
  color: string
  actionLevel: string
  description: string
  nextSteps: string[]
}> = {
  stable: {
    stageLabel: 'STABLE',
    color: '#10b981',
    actionLevel: 'Monitor',
    description: 'System operating within normal parameters',
    nextSteps: [
      'Continue routine monitoring',
      'Log baseline metrics',
      'Verify all sensors responsive',
    ],
  },
  emerging: {
    stageLabel: 'EMERGING',
    color: '#f59e0b',
    actionLevel: 'Watch',
    description: 'Early drift signals detected',
    nextSteps: [
      'Increase observation frequency',
      'Identify primary driver',
      'Prepare mitigation procedures',
    ],
  },
  approaching: {
    stageLabel: 'APPROACHING TRANSITION',
    color: '#f97316',
    actionLevel: 'Prepare',
    description: 'Transition becoming likely',
    nextSteps: [
      'Alert maintenance standby',
      'Inspect primary driver subsystem',
      'Review corrective procedures',
    ],
  },
  critical: {
    stageLabel: 'CRITICAL',
    color: '#ef4444',
    actionLevel: 'Act Now',
    description: 'System intervention required',
    nextSteps: [
      'Execute corrective action immediately',
      'Reduce system load if possible',
      'Prepare for manual override',
    ],
  },
}

// Calculate confidence score based on system metrics
export function calculateConfidenceScore(drift: number, coherence: number): number {
  // High coherence + low drift = high confidence in system state
  const coherenceFactor = coherence * 0.6
  const stabilityFactor = Math.max(0, (1 - drift) * 0.4)
  return Math.round((coherenceFactor + stabilityFactor) * 100)
}

// Classify confidence level
export function classifyConfidence(score: number): 'Low' | 'Medium' | 'High' {
  if (score >= 70) return 'High'
  if (score >= 40) return 'Medium'
  return 'Low'
}

// Calculate urgency level based on drift
export function calculateUrgencyLevel(drift: number): 'stable' | 'emerging' | 'approaching' | 'critical' {
  if (drift < 0.15) return 'stable'
  if (drift < 0.35) return 'emerging'
  if (drift < 0.6) return 'approaching'
  return 'critical'
}

// Get recommended actions for a subsystem
export function getRecommendedActions(subsystemName: string, driftContribution: number): string[] {
  const subsystemKey = subsystemName.toLowerCase()
  const mapping = subsystemActionMapping[subsystemKey]

  if (!mapping) {
    return ['Monitor subsystem closely', 'Check sensor readings']
  }

  // Return top 2-3 actions based on drift contribution
  if (driftContribution > 20) {
    return mapping.actions.slice(0, 2)
  }
  return [mapping.actions[0]]
}

// Get primary inspection points
export function getInspectionPoints(subsystemName: string): string[] {
  const subsystemKey = subsystemName.toLowerCase()
  const mapping = subsystemActionMapping[subsystemKey]
  return mapping?.inspectionPoints || []
}
