import { LiveSystem, LiveRoom } from './simulation'

export interface FacilityContext {
  activeRoomCount: number
  roomsNeedingAttention: number
  mostUrgentRoom: LiveRoom | null
  patternClusters: Array<{ label: string; rooms: string[] }>
  recentChange: string
  facilityLatency: number
  cycleContext: string
  facilityStateLabel: string
  coherenceTrend: 'weakening' | 'stable' | 'recovering'
  attentionPriority: string
  emergingIssue: string
}

export function roomStateLabel(room: LiveRoom): string {
  const d = room.driftContribution
  if (d > 0.7) return 'Critical Divergence'
  if (d > 0.4) return 'Instability Forming'
  if (d > 0.2) return 'Early Drift'
  return 'Stable'
}

export function computeFacilityContext(system: LiveSystem): FacilityContext {
  const rooms = system.rooms
  const activeRoomCount = rooms.length
  const attentionRooms = rooms.filter(r => r.status !== 'optimal')
  const roomsNeedingAttention = attentionRooms.length

  const mostUrgentRoom = rooms.length
    ? rooms.reduce((a, b) => (a.driftContribution > b.driftContribution ? a : b))
    : null

  // Pattern clusters: group by behavioralState
  const clustersMap = new Map<string, string[]>()
  rooms.forEach(r => {
    const key = r.behavioralState
    const arr = clustersMap.get(key) ?? []
    arr.push(r.shortName)
    clustersMap.set(key, arr)
  })
  const patternClusters = Array.from(clustersMap.entries())
    .filter(([, arr]) => arr.length > 1)
    .map(([label, rooms]) => ({ label, rooms }))

  // Recent change string
  let recentChange = 'No significant changes in the last 5 minutes'
  if (mostUrgentRoom) {
    if (mostUrgentRoom.driftContribution > 0.6) {
      recentChange = `${mostUrgentRoom.shortName} drift accelerated ~${system.onsetMinutes ?? 3} min ago`
    } else if (mostUrgentRoom.driftContribution > 0.35) {
      recentChange = `${mostUrgentRoom.shortName} entered instability window`
    } else if (mostUrgentRoom.lastIrrigationMinutes < 5) {
      recentChange = `${mostUrgentRoom.shortName} irrigation cycle completed`
    }
  }

  // Simulated latency with slight noise
  const facilityLatency = 0.8 + (Math.sin(system.timestamp.getTime() / 10000) + 1) * 0.3

  // Cycle context based on system timestamp
  const hour = system.timestamp.getHours()
  let cycleContext = ''
  if (hour >= 6 && hour < 12) cycleContext = 'Morning / lights on'
  else if (hour >= 12 && hour < 18) cycleContext = 'Mid-day / peak photosynthesis'
  else if (hour >= 18 && hour < 22) cycleContext = 'Evening / transition'
  else cycleContext = 'Night / recovery window'

  // Facility state label
  let facilityStateLabel = 'All systems nominal'
  if (system.severity === 'HIGH') facilityStateLabel = 'Critical regime active'
  else if (system.severity === 'ELEVATED') facilityStateLabel = 'Attention required'
  else if (system.severity === 'MODERATE') facilityStateLabel = 'Drift forming'

  // Coherence trend
  let coherenceTrend: FacilityContext['coherenceTrend'] = 'stable'
  if (system.coherence < 0.4) coherenceTrend = 'weakening'
  else if (system.structuralDrift > 0.5) coherenceTrend = 'weakening'
  else if (system.coherence > 0.8 && system.structuralDrift < 0.2) coherenceTrend = 'recovering'

  // Attention priority
  const attentionPriority = mostUrgentRoom
    ? `${mostUrgentRoom.shortName} — ${roomStateLabel(mostUrgentRoom).toLowerCase()}`
    : 'None — all rooms stable'

  // Emerging issue
  let emergingIssue = 'No emerging issues detected'
  if (mostUrgentRoom) {
    if (mostUrgentRoom.driftContribution > 0.6) {
      emergingIssue = `${mostUrgentRoom.shortName}: ${mostUrgentRoom.behavioralState}`
    } else if (patternClusters.length > 0) {
      emergingIssue = `Shared pattern: ${patternClusters[0].label} across ${patternClusters[0].rooms.join(', ')}`
    } else if (mostUrgentRoom.driftContribution > 0.3) {
      emergingIssue = `${mostUrgentRoom.shortName} showing early structural separation`
    }
  }

  return {
    activeRoomCount,
    roomsNeedingAttention,
    mostUrgentRoom,
    patternClusters,
    recentChange,
    facilityLatency,
    cycleContext,
    facilityStateLabel,
    coherenceTrend,
    attentionPriority,
    emergingIssue,
  }
}
