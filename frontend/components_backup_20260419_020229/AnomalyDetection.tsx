'use client'

import React from 'react'
import styles from './AnomalyDetection.module.css'

interface Anomaly {
  metric: string
  change: number
  severity: 'high' | 'medium'
}

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
  anomalies: Anomaly[]
  history: HistoryPoint[]
}

export default function AnomalyDetection({ anomalies, history }: Props) {
  if (anomalies.length === 0) return null

  return (
    <div className={styles.container}>
      <h2>Detected Anomalies</h2>
      <div className={styles.anomalies}>
        {anomalies.map((anom, idx) => (
          <div key={idx} className={`${styles.item} ${styles[anom.severity]}`}>
            <span className={styles.metric}>{anom.metric}</span>
            <span className={styles.change}>
              {anom.change > 0 ? '+' : ''}{anom.change.toFixed(1)}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
