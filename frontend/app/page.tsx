'use client'

import { TeslaAutopilotInterface } from '@/components/TeslaAutopilotInterface'
import { useLiveSimulation } from '@/lib/simulation'

export default function Home() {
  const { live, adaptedGrowState, isPlaying, setIsPlaying, stepScenario } = useLiveSimulation()

  const systemData = {
    drift: live.structuralDrift,
    stability: live.stability,
    coherence: live.coherence,
    confidence: live.confidenceLabel === 'HIGH' ? 0.8 : live.confidenceLabel === 'MEDIUM' ? 0.5 : 0.2,
    timestamp: live.timestamp.toISOString(),
    systemState: (live.stage as any) || 'Stable',
  }

  const rooms = live.rooms.map(r => ({
    id: r.id,
    shortName: r.shortName,
    status: r.status,
    driftContribution: r.driftContribution,
    behavioralState: r.behavioralState,
  }))

  const intelligence = {
    explanation: live.explanation,
    operatorFocus: live.operatorFocus,
    pathOutlook: live.pathOutlook,
    primaryDriver: live.primaryDriver,
  }

  return (
    <TeslaAutopilotInterface
      systemData={systemData}
      onTogglePlay={(playing) => setIsPlaying(playing)}
      rooms={rooms}
      intelligence={intelligence}
      onStepScenario={stepScenario}
      operatorZones={adaptedGrowState?.zones}
      activeEvents={adaptedGrowState?.activeEvents}
      lastChangedAt={adaptedGrowState?.lastChangedAt ?? null}
    />
  )
}
