'use client'

import { TeslaAutopilotInterface } from '@/components/TeslaAutopilotInterface'
import { useLiveSimulation } from '@/lib/simulation'

export default function Home() {
  const { live, isPlaying, setIsPlaying } = useLiveSimulation()

  const systemData = {
    drift: live.structuralDrift,
    stability: live.stability,
    coherence: live.coherence,
    confidence: live.confidenceLabel === 'HIGH' ? 0.8 : live.confidenceLabel === 'MEDIUM' ? 0.5 : 0.2,
    timestamp: live.timestamp.toISOString(),
    systemState: (live.stage as any) || 'Stable',
  }

  return (
    <TeslaAutopilotInterface
      systemData={systemData}
      onTogglePlay={(playing) => setIsPlaying(playing)}
    />
  )
}
