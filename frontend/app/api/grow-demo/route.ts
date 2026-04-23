import { NextResponse } from 'next/server'
import { loadRunnerCyclesFromDisk } from '@/lib/runnerOutputLoader'
import { adaptRunnerCyclesToGrowScenario } from '@/lib/growScenarioAdapter'
import { FALLBACK_RUNNER_CYCLES } from '@/lib/demo/fallbackRunnerCycles'

export const dynamic = 'force-dynamic'

export async function GET() {
  try {
    const cycles = await loadRunnerCyclesFromDisk()
    const scenario = adaptRunnerCyclesToGrowScenario(cycles)
    return NextResponse.json(scenario)
  } catch {
    const scenario = adaptRunnerCyclesToGrowScenario(FALLBACK_RUNNER_CYCLES)
    return NextResponse.json({ ...scenario, source: { kind: 'fallback' as const } })
  }
}

