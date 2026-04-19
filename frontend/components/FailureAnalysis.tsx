'use client'

import React from 'react'
import styles from './FailureAnalysis.module.css'
import type { Room } from './GrowOpDashboard'

type FailureMode =
  | 'hvac-failure'
  | 'sensor-drift'
  | 'humidifier-failure'
  | 'power-loss'
  | 'co2-system-failure'
  | 'none'

interface Props {
  failureMode: FailureMode
  room: Room
}

export default function FailureAnalysis({ failureMode, room }: Props) {
  const failureInfo: Record<FailureMode, any> = {
    'hvac-failure': {
      icon: '💨',
      title: 'HVAC System Failure',
      description: 'Air conditioning unit not maintaining setpoint',
      symptoms: ['Temperature rising', 'Humidity dropping', 'VPD > 2.0'],
      action: 'Inspect AC unit, check filters, verify compressor running',
      urgency: 'CRITICAL',
    },
    'humidifier-failure': {
      icon: '💧',
      title: 'Humidifier Failure',
      description: 'Humidity control system not responding',
      symptoms: ['Rapid humidity drop', 'System cannot recover'],
      action: 'Check water supply, inspect nozzles, verify pump running',
      urgency: 'HIGH',
    },
    'co2-system-failure': {
      icon: '🔬',
      title: 'CO₂ System Failure',
      description: 'CO₂ injection system malfunctioning',
      symptoms: ['CO₂ rising above target', 'Cannot reduce CO₂'],
      action: 'Check CO₂ tank level, inspect regulator, verify solenoid',
      urgency: 'MEDIUM',
    },
    'sensor-drift': {
      icon: '📡',
      title: 'Sensor Malfunction',
      description: 'Environmental sensor reading inconsistently',
      symptoms: ['Erratic readings', 'Data variance too high'],
      action: 'Verify sensor placement, check calibration, inspect wiring',
      urgency: 'HIGH',
    },
    'power-loss': {
      icon: '🔌',
      title: 'Power System Issue',
      description: 'Possible power supply or circuit issue',
      symptoms: ['Multiple systems offline', 'Simultaneous failures'],
      action: 'Check breakers, verify power distribution, test circuits',
      urgency: 'CRITICAL',
    },
    'none': {
      icon: '✓',
      title: 'System Nominal',
      description: 'All systems operating within parameters',
      symptoms: [],
      action: 'Continue monitoring',
      urgency: 'NONE',
    },
  }

  const info = failureInfo[failureMode]

  return (
    <div className={`${styles.card} ${styles[failureMode === 'none' ? 'optimal' : 'critical']}`}>
      <div className={styles.header}>
        <span className={styles.icon}>{info.icon}</span>
        <div>
          <div className={styles.title}>{info.title}</div>
          <div className={styles.urgency}>{info.urgency}</div>
        </div>
      </div>

      <p className={styles.description}>{info.description}</p>

      {info.symptoms.length > 0 && (
        <div className={styles.symptoms}>
          <div className={styles.label}>Symptoms:</div>
          {info.symptoms.map((symptom, idx) => (
            <div key={idx} className={styles.symptom}>
              • {symptom}
            </div>
          ))}
        </div>
      )}

      <div className={styles.action}>
        <div className={styles.label}>Action Required:</div>
        <div className={styles.actionText}>{info.action}</div>
      </div>
    </div>
  )
}
