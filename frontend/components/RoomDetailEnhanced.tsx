'use client'

import React, { useState, useEffect, useMemo } from 'react'
import styles from './RoomDetailEnhanced.module.css'
import type { Room } from './GrowOpDashboard'
import HistoricalTimeline from './HistoricalTimeline'
import FailureAnalysis from './FailureAnalysis'
import ActionPanel from './ActionPanel'
import YieldImpact from './YieldImpact'
import RoomComparison from './RoomComparison'
import AnomalyDetection from './AnomalyDetection'

interface HistoryPoint {
  timestamp: number
  temp: number
  humidity: number
  co2: number
  vpd: number
  ec: number
  ph: number
}

interface Props {
  room: Room
  onBack: () => void
  building: any
  allRooms: Room[]
}

type FailureMode =
  | 'hvac-failure'
  | 'sensor-drift'
  | 'humidifier-failure'
  | 'power-loss'
  | 'co2-system-failure'
  | 'none'

export default function RoomDetailEnhanced({
  room,
  onBack,
  building,
  allRooms
}: Props) {
  const [history, setHistory] = useState<HistoryPoint[]>([])
  const [playbackTime, setPlaybackTime] = useState<number | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [failureMode, setFailureMode] = useState<FailureMode>('none')
  const [yieldLoss, setYieldLoss] = useState(0)
  const [interventionTime, setInterventionTime] = useState<number | null>(null)

  // Enhanced realistic degradation simulation
  useEffect(() => {
    const startTime = Date.now()
    let pointIndex = 0

    const interval = setInterval(() => {
      setHistory((prev) => {
        const newPoint: HistoryPoint = {
          timestamp: startTime + pointIndex * 2000,
          ...generateRealisticPoint(pointIndex, room.id),
        }
        const updated = [...prev, newPoint]

        // Detect failure mode
        const detected = detectFailureMode(updated)
        if (detected !== failureMode) {
          setFailureMode(detected)
        }

        // Calculate yield loss
        const loss = calculateYieldLoss(updated, pointIndex)
        setYieldLoss(loss)

        // Check intervention time
        if (loss > 20 && !interventionTime) {
          setInterventionTime(startTime + pointIndex * 2000)
        }

        if (updated.length > 200) {
          return updated.slice(-200)
        }
        return updated
      })

      pointIndex++
    }, 2000)

    return () => clearInterval(interval)
  }, [room.id, failureMode, interventionTime])

  const generateRealisticPoint = (index: number, roomId: string): Omit<HistoryPoint, 'timestamp'> => {
    if (roomId !== 'room-4') {
      return {
        temp: 75 + (Math.random() - 0.5) * 0.8,
        humidity: 62 + (Math.random() - 0.5) * 0.8,
        co2: 1350 + (Math.random() - 0.5) * 20,
        vpd: 1.1 + (Math.random() - 0.5) * 0.05,
        ec: 1.6 + (Math.random() - 0.5) * 0.1,
        ph: 6.1 + (Math.random() - 0.5) * 0.05,
      }
    }

    // Room 4 degradation with realistic failure modes
    if (index < 20) {
      // Phase 1: Stable
      return {
        temp: 75 + (Math.random() - 0.5) * 0.8,
        humidity: 62 + (Math.random() - 0.5) * 0.8,
        co2: 1350 + (Math.random() - 0.5) * 20,
        vpd: 1.1 + (Math.random() - 0.5) * 0.05,
        ec: 1.6 + (Math.random() - 0.5) * 0.1,
        ph: 6.1 + (Math.random() - 0.5) * 0.05,
      }
    }

    if (index < 50) {
      // Phase 2: Early drift - HVAC starting to fail
      const drift = (index - 20) / 30
      return {
        temp: 75 + drift * 8 + (Math.random() - 0.4) * 1,
        humidity: 62 - drift * 12 + (Math.random() - 0.4) * 1,
        co2: 1350 + drift * 100 + (Math.random() - 0.5) * 30,
        vpd: 1.1 + drift * 0.8 + (Math.random() - 0.4) * 0.1,
        ec: 1.6 + drift * 0.3 + (Math.random() - 0.5) * 0.15,
        ph: 6.1 - drift * 0.2 + (Math.random() - 0.5) * 0.08,
      }
    }

    // Phase 3: Critical failure
    const drift = (index - 50) / 50
    return {
      temp: Math.min(90, 83 + drift * 7 + (Math.random() - 0.3) * 2),
      humidity: Math.max(30, 50 - drift * 20 + (Math.random() - 0.3) * 2),
      co2: Math.min(2000, 1450 + drift * 500 + (Math.random() - 0.5) * 50),
      vpd: Math.min(3.0, 1.9 + drift * 1.0 + (Math.random() - 0.3) * 0.2),
      ec: 1.9 + drift * 0.5 + (Math.random() - 0.5) * 0.2,
      ph: 5.9 - drift * 0.3 + (Math.random() - 0.5) * 0.1,
    }
  }

  const detectFailureMode = (hist: HistoryPoint[]): FailureMode => {
    if (hist.length < 15) return 'none'

    const recent = hist.slice(-15)
    const older = hist.slice(-30, -15)

    const tempTrend = recent[recent.length - 1].temp - older[0].temp
    const humidityTrend = recent[recent.length - 1].humidity - older[0].humidity
    const vpd = recent[recent.length - 1].vpd

    // HVAC failure: temp rising + humidity falling
    if (tempTrend > 3 && humidityTrend < -5 && vpd > 1.5) {
      return 'hvac-failure'
    }

    // Humidifier failure: humidity drops quickly
    if (humidityTrend < -8 && tempTrend < 2) {
      return 'humidifier-failure'
    }

    // CO2 system failure: CO2 rising significantly
    if (recent[recent.length - 1].co2 > 1600) {
      return 'co2-system-failure'
    }

    // Sensor drift: erratic readings
    const variance = recent.map(p => p.temp).reduce((a, b) => a + Math.abs(b - recent[0].temp), 0) / recent.length
    if (variance > 2) {
      return 'sensor-drift'
    }

    return 'none'
  }

  const calculateYieldLoss = (hist: HistoryPoint[], daysElapsed: number): number => {
    if (hist.length < 5) return 0

    const recent = hist.slice(-5)
    const avgVpd = recent.reduce((sum, p) => sum + p.vpd, 0) / recent.length
    const avgTemp = recent.reduce((sum, p) => sum + p.temp, 0) / recent.length
    const avgHumidity = recent.reduce((sum, p) => sum + p.humidity, 0) / recent.length

    let loss = 0

    // VPD impact (most critical)
    if (avgVpd > 1.4) loss += (avgVpd - 1.4) * 8
    if (avgVpd > 2.0) loss += (avgVpd - 2.0) * 15

    // Temperature impact
    if (avgTemp > 82) loss += (avgTemp - 82) * 2
    if (avgTemp > 86) loss += (avgTemp - 86) * 5

    // Humidity impact
    if (avgHumidity < 45) loss += (45 - avgHumidity) * 1.5
    if (avgHumidity < 40) loss += (40 - avgHumidity) * 3

    // Days elapsed factor - compounding
    loss = loss * (1 + daysElapsed * 0.02)

    return Math.min(100, Math.max(0, loss))
  }

  const anomalies = useMemo(() => {
    if (history.length < 2) return []
    const prev = history[history.length - 2]
    const curr = history[history.length - 1]

    const changes = []
    if (Math.abs(curr.temp - prev.temp) > 1) {
      changes.push({
        metric: 'temperature',
        change: curr.temp - prev.temp,
        severity: Math.abs(curr.temp - prev.temp) > 2 ? 'high' : 'medium',
      })
    }
    if (Math.abs(curr.humidity - prev.humidity) > 3) {
      changes.push({
        metric: 'humidity',
        change: curr.humidity - prev.humidity,
        severity: Math.abs(curr.humidity - prev.humidity) > 5 ? 'high' : 'medium',
      })
    }
    return changes
  }, [history])

  const displayHistory = playbackTime !== null
    ? history.find(h => h.timestamp <= playbackTime) || history[0]
    : history[history.length - 1]

  return (
    <div className={styles.container}>
      <div className={styles.topBar}>
        <button className={styles.backButton} onClick={onBack}>
          ← Back
        </button>
        <h1>{room.name}</h1>
        <div />
      </div>

      {/* CRITICAL METRICS + ACTIONS */}
      <div className={styles.criticalSection}>
        <div className={styles.stateCard}>
          <div className={styles.label}>Current State</div>
          <div className={styles.stateBadge + ' ' + styles[failureMode !== 'none' ? 'critical' : 'optimal']}>
            {failureMode !== 'none' ? '🔴' : '🟢'}
            {failureMode === 'none' ? 'OPTIMAL' : failureMode.toUpperCase().replace('-', ' ')}
          </div>
        </div>

        <YieldImpact yieldLoss={yieldLoss} daysInCycle={history.length / 60} />

        <FailureAnalysis failureMode={failureMode} room={room} />
      </div>

      {/* HISTORICAL TIMELINE WITH SCRUBBER */}
      <HistoricalTimeline
        history={history}
        playbackTime={playbackTime}
        onPlaybackChange={setPlaybackTime}
        isPlaying={isPlaying}
        onPlayToggle={() => setIsPlaying(!isPlaying)}
      />

      {/* ANOMALY DETECTION */}
      <AnomalyDetection anomalies={anomalies} history={history} />

      {/* RECOMMENDED ACTIONS */}
      <ActionPanel
        failureMode={failureMode}
        yieldLoss={yieldLoss}
        currentMetrics={displayHistory}
      />

      {/* ROOM COMPARISON */}
      <RoomComparison currentRoom={room} allRooms={allRooms} />
    </div>
  )
}
