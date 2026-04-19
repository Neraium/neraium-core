'use client'

import React, { useState, useEffect } from 'react'
import styles from './RoomDetail.module.css'
import type { Room } from './GrowOpDashboard'
import StateTimeline from './StateTimeline'
import DriftIndicator from './DriftIndicator'
import SensorChart from './SensorChart'

interface Props {
  room: Room
  onBack: () => void
  building: any
}

const getThresholds = (sensor: string) => {
  const thresholds: Record<string, any> = {
    temperature_f: {
      min: 68,
      max: 82,
      critical_max: 88,
      unit: '°F',
    },
    humidity_rh: {
      min: 45,
      max: 70,
      unit: '%',
    },
    co2_ppm: {
      min: 800,
      max: 1600,
      unit: 'ppm',
    },
    vpd_kpa: {
      min: 0.6,
      max: 1.4,
      critical_max: 2.0,
      unit: 'kPa',
    },
    ph: {
      min: 5.5,
      max: 6.5,
      unit: 'pH',
    },
    ec_ms: {
      min: 1.0,
      max: 2.2,
      unit: 'mS/cm',
    },
    ppfd_umol: {
      min: 400,
      max: 750,
      unit: 'µmol/m²/s',
    },
    irrigation_ml: {
      min: 100,
      max: 350,
      unit: 'mL/event',
    },
  }
  return thresholds[sensor]
}

const getSensorLabel = (key: string) => {
  const labels: Record<string, string> = {
    temperature_f: 'Temperature',
    humidity_rh: 'Humidity',
    co2_ppm: 'CO₂',
    vpd_kpa: 'VPD',
    ph: 'pH',
    ec_ms: 'EC',
    ppfd_umol: 'Light Intensity',
    irrigation_ml: 'Irrigation',
  }
  return labels[key] || key
}

const getStatusForValue = (
  key: string,
  value: number
): 'optimal' | 'warning' | 'critical' => {
  const threshold = getThresholds(key)
  if (!threshold) return 'optimal'

  if (threshold.critical_max && value > threshold.critical_max) {
    return 'critical'
  }
  if (value < threshold.min || value > threshold.max) {
    return 'warning'
  }
  return 'optimal'
}

export default function RoomDetail({ room, onBack, building }: Props) {
  const [historyData, setHistoryData] = useState<Record<string, number[]>>({})
  const [driftScore, setDriftScore] = useState(0)
  const [timeToIntervention, setTimeToIntervention] = useState('24-48 hours')

  useEffect(() => {
    const sensorKeys = Object.keys(room.sensors) as Array<
      keyof typeof room.sensors
    >
    const initialHistory: Record<string, number[]> = {}

    sensorKeys.forEach((key) => {
      initialHistory[key] = [room.sensors[key]]
    })

    setHistoryData(initialHistory)

    const interval = setInterval(() => {
      setHistoryData((prev) => {
        const updated = { ...prev }

        sensorKeys.forEach((key) => {
          const currentValue = prev[key]?.[prev[key].length - 1] || 0
          let nextValue = currentValue

          if (key === 'temperature_f' && room.id === 'room-4') {
            nextValue =
              currentValue + (Math.random() - 0.4) * 1.5 + (Math.random() - 0.5)
          } else if (key === 'humidity_rh' && room.id === 'room-4') {
            nextValue =
              currentValue + (Math.random() - 0.6) * 1.2 + (Math.random() - 0.4)
          } else {
            nextValue =
              currentValue + (Math.random() - 0.5) * 0.5 + (Math.random() - 0.35)
          }

          updated[key] = [...(prev[key] || []), parseFloat(nextValue.toFixed(2))]

          if (updated[key].length > 120) {
            updated[key] = updated[key].slice(-120)
          }
        })

        // Calculate drift score
        if (room.id === 'room-4') {
          const newDriftScore = Math.min(95, 20 + (updated['temperature_f']?.length || 0) * 0.5)
          setDriftScore(newDriftScore)
          setTimeToIntervention(newDriftScore > 70 ? '4-8 hours' : '12-24 hours')
        }

        return updated
      })
    }, 2000)

    return () => clearInterval(interval)
  }, [room.id])

  const keySignals = ['temperature_f', 'humidity_rh', 'co2_ppm', 'vpd_kpa']
  const getSensorEntries = () => {
    return Object.entries(room.sensors).filter(([key]) =>
      keySignals.includes(key)
    ) as Array<[string, number]>
  }

  return (
    <div className={styles.container}>
      <div className={styles.topBar}>
        <button className={styles.backButton} onClick={onBack}>
          ← Back
        </button>
        <h1>{room.name}</h1>
        <div />
      </div>

      {/* ROOM STATE & INTERVENTION WINDOW */}
      <div className={styles.criticalSection}>
        <div className={styles.stateCard}>
          <div className={styles.stateLabel}>Current State</div>
          <div className={`${styles.stateBadge} ${styles[room.status]}`}>
            {room.status === 'critical' ? '🔴' : room.status === 'warning' ? '🟡' : '🟢'}
            {room.status.toUpperCase()}
          </div>
          <div className={styles.stageText}>{room.stage}</div>
        </div>

        <DriftIndicator driftScore={driftScore} room={room} />

        <div className={styles.interventionCard}>
          <div className={styles.interventionLabel}>Action Window</div>
          <div className={styles.interventionTime}>{timeToIntervention}</div>
          <div className={styles.interventionSubtext}>
            {driftScore > 70 ? '⚠️ Limited time to intervene' : '✓ Window still open'}
          </div>
        </div>
      </div>

      {/* EVOLUTION TIMELINE */}
      <StateTimeline room={room} historyData={historyData} />

      {/* KEY SIGNALS - FOCUSED VIEW */}
      <div className={styles.signalsSection}>
        <h2>Key Climate Signals</h2>
        <div className={styles.chartsGrid}>
          {getSensorEntries().map(([key, value]) => {
            const threshold = getThresholds(key)
            const status = getStatusForValue(key, value)
            const history = historyData[key] || [value]

            return (
              <div key={key} className={styles.chartContainer}>
                <SensorChart
                  label={getSensorLabel(key)}
                  value={value}
                  unit={threshold?.unit || ''}
                  status={status}
                  history={history}
                  threshold={threshold}
                />
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

