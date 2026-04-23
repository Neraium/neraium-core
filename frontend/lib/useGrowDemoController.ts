'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import type { GrowScenario, GrowScenarioFrame, GrowZoneId, ZoneSeverity } from '@/lib/growScenarioAdapter'

export interface DemoOperatorView {
  systemData: {
    drift: number
    stability: number
    coherence: number
    confidence: number
    timestamp: string
    systemState: 'Stable' | 'Drift forming' | 'Instability forming' | 'Critical'
  }
  rooms: Array<{
    id: string
    shortName: string
    status: 'optimal' | 'warning' | 'critical'
    driftContribution: number
    behavioralState: string
  }>
  intelligence: {
    explanation: string
    operatorFocus: string
    pathOutlook: string
    primaryDriver: string
  }
}

function zoneIdToRoomId(zone: GrowZoneId): string {
  return zone.toLowerCase().replace(/[^a-z0-9]+/g, '_')
}

function severityToRoomStatus(sev: ZoneSeverity): 'optimal' | 'warning' | 'critical' {
  if (sev === 'red') return 'critical'
  if (sev === 'yellow') return 'warning'
  return 'optimal'
}

function deriveSystemState(frame: GrowScenarioFrame): DemoOperatorView['systemData']['systemState'] {
  if (frame.systemSeverity === 'red') return 'Critical'
  if (frame.systemSeverity === 'yellow') return 'Drift forming'
  return 'Stable'
}

function frameToOperatorView(frame: GrowScenarioFrame, scenario: GrowScenario): DemoOperatorView {
  const drift = frame.details.drift ?? 0
  const instability = frame.details.instability ?? 0
  const stability = Math.max(0, Math.min(1, 1 - instability))
  const coherence = Math.max(0, Math.min(1, 1 - drift * 0.85))
  const confidence = 0.65

  const timestamp = new Date().toISOString()

  const rooms = scenario.zones.map(zone => {
    const sev = frame.zones[zone]
    const contribution =
      sev === 'red' ? Math.max(0.55, drift) :
      sev === 'yellow' ? Math.max(0.32, drift * 0.7) :
      Math.max(0.05, drift * 0.15)
    return {
      id: zoneIdToRoomId(zone),
      shortName: zone,
      status: severityToRoomStatus(sev),
      driftContribution: contribution,
      behavioralState: frame.details.driver || frame.details.trend || frame.details.phase || '',
    }
  })

  return {
    systemData: {
      drift,
      stability,
      coherence,
      confidence,
      timestamp,
      systemState: deriveSystemState(frame),
    },
    rooms,
    intelligence: {
      explanation: frame.details.operatorMessage || '',
      operatorFocus: frame.issue,
      pathOutlook: frame.progression,
      primaryDriver: frame.details.driver || '',
    },
  }
}

export function useGrowDemoController(opts?: { tickMs?: number; autoPlay?: boolean }) {
  const tickMs = opts?.tickMs ?? 850
  const autoPlay = opts?.autoPlay ?? true

  const [scenario, setScenario] = useState<GrowScenario | null>(null)
  const [isPlaying, setIsPlaying] = useState(autoPlay)
  const [index, setIndex] = useState(0)

  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      try {
        const res = await fetch('/api/grow-demo', { cache: 'no-store' })
        const json = (await res.json()) as GrowScenario
        if (cancelled) return
        setScenario(json)
        setIndex(0)
      } catch {
        // Scenario route itself already falls back; this is last resort.
      }
    }
    load()
    return () => {
      cancelled = true
    }
  }, [])

  const frame: GrowScenarioFrame | null = useMemo(() => {
    if (!scenario || scenario.frames.length === 0) return null
    return scenario.frames[Math.max(0, Math.min(index, scenario.frames.length - 1))]!
  }, [scenario, index])

  const operatorView: DemoOperatorView | null = useMemo(() => {
    if (!scenario || !frame) return null
    return frameToOperatorView(frame, scenario)
  }, [scenario, frame])

  useEffect(() => {
    if (!scenario || scenario.frames.length === 0) return
    if (!isPlaying) {
      if (timerRef.current) clearInterval(timerRef.current)
      timerRef.current = null
      return
    }

    if (timerRef.current) clearInterval(timerRef.current)
    timerRef.current = setInterval(() => {
      setIndex(prev => {
        const next = prev + 1
        if (next >= scenario.frames.length) return scenario.frames.length - 1
        return next
      })
    }, tickMs)

    return () => {
      if (timerRef.current) clearInterval(timerRef.current)
      timerRef.current = null
    }
  }, [isPlaying, scenario, tickMs])

  const controls = useMemo(() => {
    const play = () => setIsPlaying(true)
    const pause = () => setIsPlaying(false)
    const reset = () => {
      setIsPlaying(false)
      setIndex(0)
    }
    const step = (delta: number) => {
      setIsPlaying(false)
      setIndex(prev => {
        if (!scenario) return prev
        const next = Math.max(0, Math.min(prev + delta, scenario.frames.length - 1))
        return next
      })
    }
    return { play, pause, reset, step }
  }, [scenario])

  return {
    scenario,
    frame,
    operatorView,
    index,
    isPlaying,
    ...controls,
  }
}
