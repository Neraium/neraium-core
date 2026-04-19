'use client'

import React, { useState, useEffect } from 'react'
import styles from './RoomDetail.module.css'
import type { Room } from './GrowOpDashboard'
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
    temperature_f: 'Canopy Temperature',
    humidity_rh: 'Relative Humidity',
    co2_ppm: 'CO₂ Concentration',
    vpd_kpa: 'Vapor Pressure Deficit',
    ph: 'Reservoir pH',
    ec_ms: 'Electrical Conductivity',
    ppfd_umol: 'Light Intensity (PPFD)',
    irrigation_ml: 'Irrigation Volume',
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
  const [timeRange, setTimeRange] = useState<'1h' | '6h' | '24h'>('1h')

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

        return updated
      })
    }, 2000)

    return () => clearInterval(interval)
  }, [room.id])

  const getSensorEntries = () => {
    return Object.entries(room.sensors) as Array<
      [string, number]
    >
  }

  return (
    <div className={styles.container}>
      <div className={styles.topBar}>
        <button className={styles.backButton} onClick={onBack}>
          ← Back to Overview
        </button>
        <h1>{room.name}</h1>
        <div />
      </div>

      <div className={styles.infoBar}>
        <div className={styles.info}>
          <span>Floor {room.floor}</span>
          <span>•</span>
          <span>{room.zone} Zone</span>
          <span>•</span>
          <span>{room.stage}</span>
          <span>•</span>
          <span>
            {room.plantCount}/{room.capacity} plants
          </span>
        </div>

        <div className={styles.timeRangeSelector}>
          {(['1h', '6h', '24h'] as const).map((range) => (
            <button
              key={range}
              className={`${styles.timeButton} ${timeRange === range ? styles.active : ''}`}
              onClick={() => setTimeRange(range)}
            >
              {range}
            </button>
          ))}
        </div>
      </div>

      <div className={styles.chartsGrid}>
        {getSensorEntries().map(([key, value]) => {
          if (key === 'ph' && room.id === 'room-5') return null
          if (key === 'ec_ms' && room.id === 'room-5') return null
          if (key === 'ppfd_umol' && room.id === 'room-5') return null
          if (key === 'irrigation_ml' && room.id === 'room-5') return null

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

      <div className={styles.metricsTable}>
        <h2>Current Sensor Readings</h2>
        <table>
          <thead>
            <tr>
              <th>Sensor</th>
              <th>Current Value</th>
              <th>Status</th>
              <th>Range</th>
            </tr>
          </thead>
          <tbody>
            {getSensorEntries().map(([key, value]) => {
              const threshold = getThresholds(key)
              const status = getStatusForValue(key, value)
              if (threshold === undefined || threshold.unit === undefined)
                return null

              return (
                <tr key={key}>
                  <td className={styles.sensorName}>{getSensorLabel(key)}</td>
                  <td className={styles.value}>
                    {value.toFixed(key === 'temperature_f' || key === 'humidity_rh' ? 1 : 2)}{' '}
                    {threshold.unit}
                  </td>
                  <td>
                    <span
                      className={`${styles.statusBadge} ${styles[status]}`}
                    >
                      {status.charAt(0).toUpperCase() + status.slice(1)}
                    </span>
                  </td>
                  <td className={styles.range}>
                    {threshold.min} - {threshold.max}{' '}
                    {threshold.critical_max && `(critical: >${threshold.critical_max})`}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
