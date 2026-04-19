'use client'

import React, { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { TopBar } from './TopBar'
import { RoomStateModel } from './RoomStateModel'
import { IntelligenceRail } from './IntelligenceRail'
import { SubsystemCard } from './SubsystemCard'
import { StateEvolutionTimeline } from './StateEvolutionTimeline'

interface RoomData {
  status: 'stable' | 'drift' | 'instability' | 'critical'
  driftScore: number
  temperature: number
  humidity: number
  co2: number
  light: number
  waterEc: number
  plantHealth: number
}

interface Subsystem {
  name: string
  status: 'optimal' | 'warning' | 'critical'
  contribution: number
  value: number
  unit: string
  trend: 'up' | 'down' | 'stable'
  sparklineData: number[]
}

// Mock data generator
const generateMockData = (degradationFactor: number = 0): RoomData & { subsystems: Subsystem[] } => {
  const baseTemp = 24
  const baseHumidity = 60
  const baseCo2 = 800
  const baseLight = 900
  const baseWaterEc = 1.2

  const tempVariation = degradationFactor * 3
  const humidityVariation = degradationFactor * 8
  const co2Variation = degradationFactor * 150
  const lightVariation = degradationFactor * 200
  const ecVariation = degradationFactor * 0.4

  const temperature = baseTemp + tempVariation + (Math.random() - 0.5) * 1
  const humidity = Math.max(20, baseHumidity - humidityVariation + (Math.random() - 0.5) * 3)
  const co2 = baseCo2 + co2Variation + (Math.random() - 0.5) * 50
  const light = Math.max(100, baseLight - lightVariation + (Math.random() - 0.5) * 50)
  const waterEc = baseWaterEc + ecVariation + (Math.random() - 0.5) * 0.1
  const plantHealth = Math.max(0, 100 - degradationFactor * 50 + (Math.random() - 0.5) * 5)

  // Calculate drift score based on how far readings are from optimal
  const tempDrift = Math.abs(temperature - 24) / 5
  const humidityDrift = Math.abs(humidity - 60) / 20
  const co2Drift = Math.abs(co2 - 800) / 500
  const lightDrift = Math.abs(light - 900) / 500
  const plantDrift = (100 - plantHealth) / 100

  const driftScore = Math.min(1, (tempDrift + humidityDrift + co2Drift + lightDrift + plantDrift) / 5 + degradationFactor)

  // Determine status
  let status: 'stable' | 'drift' | 'instability' | 'critical' = 'stable'
  if (driftScore > 0.7) status = 'critical'
  else if (driftScore > 0.5) status = 'instability'
  else if (driftScore > 0.25) status = 'drift'

  // Subsystems
  const subsystems: Subsystem[] = [
    {
      name: 'Climate',
      status: Math.abs(temperature - 24) > 2 ? 'warning' : 'optimal',
      contribution: Math.abs(temperature - 24) * 10,
      value: Math.round(temperature * 10) / 10,
      unit: '°C',
      trend: temperature > 24.5 ? 'up' : temperature < 23.5 ? 'down' : 'stable',
      sparklineData: [24, 24.2, 24.5, 24.8, 24.6, 24.3, 24],
    },
    {
      name: 'Humidity',
      status: humidity < 50 || humidity > 70 ? 'warning' : 'optimal',
      contribution: Math.abs(humidity - 60) * 2,
      value: Math.round(humidity),
      unit: '%',
      trend: humidity > 62 ? 'up' : humidity < 58 ? 'down' : 'stable',
      sparklineData: [58, 59, 61, 62, 61, 60, 59],
    },
    {
      name: 'CO₂',
      status: co2 > 1000 ? 'warning' : 'optimal',
      contribution: Math.abs(co2 - 800) / 10,
      value: Math.round(co2),
      unit: 'ppm',
      trend: co2 > 850 ? 'up' : co2 < 750 ? 'down' : 'stable',
      sparklineData: [800, 820, 850, 880, 850, 820, 800],
    },
    {
      name: 'Lighting',
      status: light < 600 ? 'warning' : 'optimal',
      contribution: Math.max(0, (900 - light) / 5),
      value: Math.round(light),
      unit: 'μmol',
      trend: light > 950 ? 'up' : light < 850 ? 'down' : 'stable',
      sparklineData: [900, 920, 950, 920, 900, 880, 900],
    },
    {
      name: 'Irrigation',
      status: waterEc > 1.4 || waterEc < 1.0 ? 'warning' : 'optimal',
      contribution: Math.abs(waterEc - 1.2) * 20,
      value: Math.round(waterEc * 100) / 100,
      unit: 'dS/m',
      trend: waterEc > 1.25 ? 'up' : waterEc < 1.15 ? 'down' : 'stable',
      sparklineData: [1.2, 1.22, 1.25, 1.23, 1.2, 1.18, 1.2],
    },
    {
      name: 'Plant Layer',
      status: plantHealth < 60 ? 'critical' : plantHealth < 80 ? 'warning' : 'optimal',
      contribution: (100 - plantHealth) / 2,
      value: Math.round(plantHealth),
      unit: '%',
      trend: plantHealth > 85 ? 'up' : plantHealth < 75 ? 'down' : 'stable',
      sparklineData: [95, 93, 90, 88, 85, 80, 75],
    },
  ]

  return {
    status,
    driftScore,
    temperature,
    humidity,
    co2,
    light,
    waterEc,
    plantHealth,
    subsystems,
  }
}

export const LouddpaxRoomIntelligence: React.FC = () => {
  const [roomData, setRoomData] = useState<RoomData & { subsystems: Subsystem[] }>(
    generateMockData(0)
  )
  const [degradationFactor, setDegradationFactor] = useState(0)
  const [timelinePoints, setTimelinePoints] = useState([
    {
      label: 'Baseline Established',
      timestamp: '2 days ago',
      status: 'baseline' as const,
      isCurrent: false,
    },
    {
      label: 'Early Drift Detected',
      timestamp: '12 hours ago',
      status: 'forming' as const,
      isCurrent: false,
    },
    {
      label: 'Instability Forming',
      timestamp: '2 hours ago',
      status: 'advanced' as const,
      isCurrent: true,
    },
  ])

  useEffect(() => {
    const interval = setInterval(() => {
      setDegradationFactor((prev) => {
        const next = prev + 0.01
        return next > 1 ? 0 : next
      })
    }, 2000)

    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    setRoomData(generateMockData(degradationFactor))
  }, [degradationFactor])

  const topContributors = roomData.subsystems
    .sort((a, b) => b.contribution - a.contribution)
    .slice(0, 3)
    .map((s) => ({
      name: s.name,
      percentage: s.contribution,
      status: s.status,
    }))

  const explanations = {
    stable: 'System operating within nominal parameters. All subsystems functioning optimally with minimal drift.',
    drift: 'Early environmental drift detected. Humidity trending down and temperature slight elevation. Recommend monitoring.',
    instability: 'Multiple drift signals accumulating. Climate control showing strain. Irrigation EC rising beyond target.',
    critical: 'Critical state forming. Plant health declining rapidly. Immediate intervention required.',
  }

  const stateLabels = {
    stable: 'Stable',
    drift: 'Early Drift',
    instability: 'Instability',
    critical: 'Critical',
  }

  return (
    <div className="flex flex-col h-screen bg-louddpax-bg text-louddpax-text overflow-hidden">
      {/* Top Bar */}
      <TopBar
        facilityName="Highland Facility"
        roomName="Room A-12"
        growthPhase="Week 4 Flower"
        isLive={true}
        globalHealthScore={Math.round((1 - roomData.driftScore) * 100)}
      />

      {/* Main Content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Center: Room State Model */}
        <div className="flex-1 flex flex-col bg-louddpax-bg p-8 gap-6 overflow-y-auto">
          <RoomStateModel roomState={roomData} subsystems={roomData.subsystems} />

          {/* State Evolution Timeline */}
          <StateEvolutionTimeline points={timelinePoints} />

          {/* Subsystems Grid */}
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            {roomData.subsystems.map((subsystem) => (
              <SubsystemCard
                key={subsystem.name}
                name={subsystem.name}
                keyValue={subsystem.value}
                unit={subsystem.unit}
                driftContribution={subsystem.contribution}
                trend={subsystem.trend}
                sparklineData={subsystem.sparklineData}
                status={subsystem.status}
              />
            ))}
          </div>

          {/* Debug Controls */}
          <div className="mt-8 px-6 py-4 bg-louddpax-surface rounded border border-louddpax-border/50">
            <p className="text-xs text-louddpax-muted tracking-widest uppercase mb-3">
              Simulation Controls
            </p>
            <div className="flex items-center gap-4">
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={degradationFactor}
                onChange={(e) => setDegradationFactor(parseFloat(e.target.value))}
                className="flex-1 h-2 bg-louddpax-border rounded-lg appearance-none cursor-pointer accent-louddpax-primary"
              />
              <span className="text-sm text-louddpax-text font-mono">
                {Math.round(degradationFactor * 100)}%
              </span>
            </div>
          </div>

          {/* Branded Footer */}
          <div className="mt-8 px-6 py-4 bg-gradient-to-r from-louddpax-primary/5 to-louddpax-primary/0 rounded border border-louddpax-primary/20">
            <div className="flex items-center justify-center gap-4">
              <div className="flex items-center gap-2">
                <div className="w-1 h-1 rounded-full bg-louddpax-primary" />
                <span className="text-xs text-louddpax-muted tracking-widest">LOUDDPAX</span>
              </div>
              <div className="w-px h-4 bg-louddpax-border/30" />
              <div className="flex items-center gap-2">
                <span className="text-xs text-louddpax-muted tracking-widest">Powered by</span>
                <span className="text-xs font-bold text-louddpax-primary tracking-widest">NERAIUM</span>
              </div>
            </div>
          </div>
        </div>

        {/* Right: Intelligence Rail */}
        <IntelligenceRail
          stateLabel={stateLabels[roomData.status]}
          driftScore={roomData.driftScore}
          confidence={0.92}
          driftOnsetTime={degradationFactor > 0.2 ? '2 hours 18 min ago' : undefined}
          topContributors={topContributors}
          explanation={explanations[roomData.status]}
        />
      </div>
    </div>
  )
}
