/**
 * Integration layer: maps gate decision + records → UI state.
 *
 * Core responsibility: transform decision engine output into decision-centric UI state.
 * No internal variables exposed. Only interpreted labels and decision output.
 */

export enum Severity {
  LOW = "LOW",
  MODERATE = "MODERATE",
  ELEVATED = "ELEVATED",
  HIGH = "HIGH",
}

export enum Trajectory {
  STABLE = "Stable",
  DEGRADING = "Degrading",
  UNCERTAIN = "Uncertain",
}

export enum DegradationStage {
  BASELINE = "baseline",
  EARLY_SHIFT = "early_shift",
  EMERGING = "emerging",
  PERSISTENT = "persistent",
  ACCELERATED = "accelerated",
  FAILURE_APPROACH = "failure_approach",
}

export enum ActionHorizon {
  NOW = "now",
  SOON = "soon",
  WATCHLIST = "watchlist",
}

export interface Point3D {
  x: number
  y: number
  z: number
}

export interface TrailPoint {
  position: Point3D
  timestamp: string
  frameIdx: number
}

export interface PatternInsight {
  tier: "weak" | "moderate" | "strong"
  outcomeType: string
  influenceSummary: string
}

export interface StatusHeader {
  degradationStage: DegradationStage
  severity: Severity
  trajectory: Trajectory
  confidence: "LOW" | "MEDIUM" | "HIGH"
}

export interface ActionPanel {
  horizon: ActionHorizon
  primaryAction: string
  urgencyLevel: number // 1-5
  decisionMomentum?: "emerging" | "establishing" | "strengthening" // Perceived strength of commitment
  commitmentScore?: number // [0, 1] strength of decision commitment
}

export interface TetrahedronState {
  currentPosition: Point3D
  severityScalar: number // 0-1 for coloring
  trailPoints: TrailPoint[]
  velocity: [number, number, number]
  nearestVertex: string
}

export interface DecisionTrace {
  primaryFactor: string
  secondaryFactors: string[]
  confidenceRationale: string
  patternInsight?: PatternInsight
}

export interface TimelineStage {
  stage: DegradationStage
  label: string
  isCurrent: boolean
}

export interface SystemTimeline {
  stages: TimelineStage[]
  currentIndex: number
}

export interface ChartDataPoint {
  timestamp: string
  driftScore: number
  frameIdx: number
}

export interface DriftChart {
  dataPoints: ChartDataPoint[]
  detectionThreshold: number
  currentFrameIdx: number
}

export interface DecisionUIState {
  statusHeader: StatusHeader
  actionPanel: ActionPanel
  tetrahedron: TetrahedronState
  decisionTrace: DecisionTrace
  timeline: SystemTimeline
  driftChart: DriftChart
  timestamp: string
}

function clamp(value: number, low: number = 0.0, high: number = 1.0): number {
  return Math.max(low, Math.min(high, value))
}

function computeDegradationStage(records: Record<string, unknown>[]): DegradationStage {
  if (records.length === 0) return DegradationStage.BASELINE

  const latest = records[records.length - 1]
  const drift = clamp(Number(latest?.structural_drift_score ?? 0), 0, 1)
  const persistenceMinutes = Number(latest?.persistence_minutes ?? 0)

  if (drift < 0.25) return DegradationStage.BASELINE
  if (drift < 0.4) return DegradationStage.EARLY_SHIFT
  if (drift < 0.55) return DegradationStage.EMERGING
  if (drift < 0.7 && persistenceMinutes < 120) return DegradationStage.PERSISTENT
  if (drift < 0.85) return DegradationStage.ACCELERATED
  return DegradationStage.FAILURE_APPROACH
}

function computeSeverity(
  gateDecision: Record<string, unknown>,
  driftScore: number,
  stabilityScore: number
): Severity {
  const confidenceLabel = String(gateDecision?.confidence_label ?? "LOW").toUpperCase()

  const instability = 1.0 - clamp(stabilityScore)
  const maxMetric = Math.max(driftScore, instability)

  if (confidenceLabel === "HIGH" && maxMetric >= 0.6) return Severity.HIGH
  if (confidenceLabel === "HIGH" || maxMetric >= 0.5) return Severity.ELEVATED
  if (confidenceLabel === "MEDIUM" || maxMetric >= 0.3) return Severity.MODERATE
  return Severity.LOW
}

function computeTrajectory(records: Record<string, unknown>[]): Trajectory {
  if (records.length < 2) return Trajectory.STABLE

  const latest = records[records.length - 1]
  const previous = records[records.length - 2]

  const deltaDrift =
    Number(latest?.structural_drift_score ?? 0) - Number(previous?.structural_drift_score ?? 0)
  const deltaStability =
    Number(latest?.relational_stability_score ?? 1) - Number(previous?.relational_stability_score ?? 1)

  if (deltaDrift > 0.08 || deltaStability < -0.08) return Trajectory.DEGRADING
  if (Math.abs(deltaDrift) > 0.03 || Math.abs(deltaStability) > 0.03) return Trajectory.UNCERTAIN
  return Trajectory.STABLE
}

function computeTetrahedronPosition(
  gateDecision: Record<string, unknown>,
  latestRecord: Record<string, unknown>
): Point3D {
  /**
   * Dimensions:
   * - x: severity (0=center=low, 1=edge=high)
   * - y: confidence (0=uncertain, 1=certain)
   * - z: trajectory rate (0=stable, 1=rapidly changing)
   */

  // X-axis: severity
  const confidenceLabel = String(gateDecision?.confidence_label ?? "LOW").toUpperCase()
  const drift = clamp(Number(latestRecord?.structural_drift_score ?? 0))

  const severityX =
    confidenceLabel === "HIGH" ? 0.8 : confidenceLabel === "MEDIUM" ? 0.5 : 0.2

  // Y-axis: confidence in the decision
  const confidenceY =
    confidenceLabel === "HIGH" ? 0.85 : confidenceLabel === "MEDIUM" ? 0.5 : 0.2

  // Z-axis: rate of change
  const persistenceMinutes = Number(latestRecord?.persistence_minutes ?? 0)
  const deltaDrift = Number(latestRecord?.delta_drift ?? 0)
  const trajectoryZ = Math.min(1.0, Math.abs(deltaDrift) * 2.0 + persistenceMinutes / 300.0)

  return {
    x: clamp(severityX),
    y: clamp(confidenceY),
    z: clamp(trajectoryZ),
  }
}

function nearestTetrahedronVertex(position: Point3D): string {
  const vertices: Record<string, [number, number, number]> = {
    STRUCTURAL: [1.0, 1.0, 1.0],
    RELATIONAL: [1.0, -1.0, -1.0],
    TEMPORAL: [-1.0, -1.0, 1.0],
    AUTHORITY: [-1.0, 1.0, -1.0],
  }

  let minDistance = Infinity
  let nearest = "STRUCTURAL"

  for (const [vertexName, vertexPos] of Object.entries(vertices)) {
    const distance = Math.sqrt(
      (position.x - vertexPos[0]) ** 2 + (position.y - vertexPos[1]) ** 2 + (position.z - vertexPos[2]) ** 2
    )
    if (distance < minDistance) {
      minDistance = distance
      nearest = vertexName
    }
  }

  return nearest
}

function extractPrimaryFactor(
  gateDecision: Record<string, unknown>,
  latestRecord: Record<string, unknown>
): string {
  const persistence = Number(latestRecord?.persistence_minutes ?? 0)
  const drift = Number(latestRecord?.structural_drift_score ?? 0)
  const stability = Number(latestRecord?.relational_stability_score ?? 1)

  if (persistence > 60) return `High severity persisted ${Math.round(persistence)} minutes`
  if (drift >= 0.6) return `Structural drift reached ${Math.round(drift * 100)}%`
  if (stability < 0.4) return `Relational stability dropped to ${Math.round(stability * 100)}%`
  return String(gateDecision?.explanation ?? "System state change detected")
}

function extractSecondaryFactors(
  gateDecision: Record<string, unknown>,
  latestRecord: Record<string, unknown>
): string[] {
  const factors: string[] = []

  const corroborating = Number(latestRecord?.corroborating_signal_count ?? 0)
  if (corroborating > 0) {
    factors.push(`${Math.round(corroborating)} corroborating signals`)
  }

  const deltaDrift = Number(latestRecord?.delta_drift ?? 0)
  if (Math.abs(deltaDrift) > 0.08) {
    const direction = deltaDrift > 0 ? "increasing" : "decreasing"
    factors.push(`Drift ${direction} by ${Math.round(Math.abs(deltaDrift) * 100)}%`)
  }

  return factors.slice(0, 2)
}

function extractPatternInsight(gateDecision: Record<string, unknown>): PatternInsight | undefined {
  const pattern = gateDecision?.pattern_match as Record<string, unknown> | undefined
  if (!pattern) return undefined

  const tier = String(pattern?.tier ?? "weak").toLowerCase()
  if (tier !== "moderate" && tier !== "strong") return undefined

  return {
    tier: tier as "moderate" | "strong",
    outcomeType: String(pattern?.outcome_type ?? "unknown"),
    influenceSummary: String(pattern?.influence_summary ?? ""),
  }
}

function buildTrail(
  records: Record<string, unknown>[],
  gateDecision: Record<string, unknown>,
  limit: number = 20
): TrailPoint[] {
  return records.slice(-limit).map((record, idx) => {
    const position = computeTetrahedronPosition(gateDecision, record)
    return {
      position,
      timestamp: String(record?.timestamp ?? ""),
      frameIdx: idx,
    }
  })
}

function buildTimeline(records: Record<string, unknown>[]): SystemTimeline {
  const stages: TimelineStage[] = []
  const currentStage = computeDegradationStage(records)
  const stageValues = Object.values(DegradationStage)

  for (let i = 0; i < stageValues.length; i++) {
    const stage = stageValues[i]
    stages.push({
      stage: stage,
      label: stage.replace(/_/g, " ").toUpperCase(),
      isCurrent: stage === currentStage,
    })
  }

  return {
    stages,
    currentIndex: stageValues.indexOf(currentStage),
  }
}

function buildDriftChart(records: Record<string, unknown>[]): DriftChart {
  const dataPoints: ChartDataPoint[] = records.slice(-24).map((record, idx) => ({
    timestamp: String(record?.timestamp ?? ""),
    driftScore: clamp(Number(record?.structural_drift_score ?? 0)),
    frameIdx: idx,
  }))

  return {
    dataPoints,
    detectionThreshold: 0.2,
    currentFrameIdx: dataPoints.length - 1,
  }
}

export function transformDecisionToUIState(
  gateDecision: Record<string, unknown>,
  records: Record<string, unknown>[]
): DecisionUIState {
  /**
   * Main entry point: transform decision + records → complete UI state.
   */
  if (records.length === 0) {
    throw new Error("Records array cannot be empty")
  }

  const latest = records[records.length - 1]

  // Extract base metrics
  const driftScore = clamp(Number(latest?.structural_drift_score ?? 0))
  const stabilityScore = clamp(Number(latest?.relational_stability_score ?? 1))
  const timestamp = String(latest?.timestamp ?? "")

  // Build status header
  const degradationStage = computeDegradationStage(records)
  const severity = computeSeverity(gateDecision, driftScore, stabilityScore)
  const trajectory = computeTrajectory(records)
  const confidenceLabel = (String(gateDecision?.confidence_label ?? "LOW").toUpperCase()) as
    | "LOW"
    | "MEDIUM"
    | "HIGH"

  const statusHeader: StatusHeader = {
    degradationStage,
    severity,
    trajectory,
    confidence: confidenceLabel,
  }

  // Build action panel
  const isAdmitted = String(gateDecision?.decision ?? "SUPPRESS").toUpperCase() === "ADMIT"
  const isHighSeverity = severity === Severity.HIGH || severity === Severity.ELEVATED

  let horizon: ActionHorizon
  let urgency: number

  if (isAdmitted && isHighSeverity) {
    horizon = ActionHorizon.NOW
    urgency = 5
  } else if (isAdmitted) {
    horizon = ActionHorizon.SOON
    urgency = 3
  } else {
    horizon = ActionHorizon.WATCHLIST
    urgency = 1
  }

  const primaryAction = String(gateDecision?.explanation ?? "Monitor system state")

  // Extract decision commitment metrics from operational_recommendation
  const operationalRec = gateDecision?.operational_recommendation as Record<string, unknown> | undefined
  const decisionMomentum = String(operationalRec?.decision_momentum ?? "emerging").toLowerCase() as
    | "emerging"
    | "establishing"
    | "strengthening"
  const commitmentScore = clamp(Number(operationalRec?.decision_commitment_score ?? 0.3))

  const actionPanel: ActionPanel = {
    horizon,
    primaryAction,
    urgencyLevel: urgency,
    decisionMomentum,
    commitmentScore,
  }

  // Build tetrahedron state
  const currentPosition = computeTetrahedronPosition(gateDecision, latest)
  const velocity: [number, number, number] = [
    Math.abs(Number(latest?.delta_drift ?? 0)),
    Math.abs(Number(latest?.delta_stability ?? 0)),
    Math.abs(Number(latest?.persistence_minutes ?? 0) / 100.0),
  ]

  const trailPoints = buildTrail(records, gateDecision)
  const severityScalars: Record<Severity, number> = {
    [Severity.LOW]: 0.2,
    [Severity.MODERATE]: 0.5,
    [Severity.ELEVATED]: 0.75,
    [Severity.HIGH]: 0.95,
  }
  const severityScalar = severityScalars[severity]
  const nearestVertex = nearestTetrahedronVertex(currentPosition)

  const tetrahedron: TetrahedronState = {
    currentPosition,
    severityScalar,
    trailPoints,
    velocity,
    nearestVertex,
  }

  // Build decision trace
  const primaryFactor = extractPrimaryFactor(gateDecision, latest)
  const secondaryFactors = extractSecondaryFactors(gateDecision, latest)
  const confidenceRationale = `Confidence level: ${confidenceLabel}`
  const patternInsight = extractPatternInsight(gateDecision)

  const decisionTrace: DecisionTrace = {
    primaryFactor,
    secondaryFactors,
    confidenceRationale,
    patternInsight,
  }

  // Build timeline
  const timeline = buildTimeline(records)

  // Build drift chart
  const driftChart = buildDriftChart(records)

  return {
    statusHeader,
    actionPanel,
    tetrahedron,
    decisionTrace,
    timeline,
    driftChart,
    timestamp,
  }
}
