'use client'

import React from 'react'
import styles from './RoomCard.module.css'
import type { Room } from './GrowOpDashboard'

interface Props {
  room: Room
  onClick: () => void
}

const getStatusColor = (status: string) => {
  switch (status) {
    case 'optimal':
      return '#10b981'
    case 'warning':
      return '#f59e0b'
    case 'critical':
      return '#ef4444'
    default:
      return '#6b7280'
  }
}

export default function RoomCard({ room, onClick }: Props) {
  const statusDisplay =
    room.status.charAt(0).toUpperCase() + room.status.slice(1)

  return (
    <div
      className={styles.card}
      onClick={onClick}
      style={{
        borderLeftColor: getStatusColor(room.status),
      }}
    >
      <div className={styles.header}>
        <div>
          <h3>{room.name}</h3>
          <p className={styles.location}>
            Floor {room.floor} • {room.zone}
          </p>
        </div>
        <div
          className={styles.statusBadge}
          style={{ backgroundColor: getStatusColor(room.status) }}
        >
          {statusDisplay}
        </div>
      </div>

      <div className={styles.stage}>{room.stage}</div>

      <div className={styles.metrics}>
        <div className={styles.metric}>
          <span className={styles.label}>Plants</span>
          <span className={styles.value}>
            {room.plantCount}/{room.capacity}
          </span>
        </div>
        <div className={styles.metric}>
          <span className={styles.label}>Temp</span>
          <span className={styles.value}>{room.sensors.temperature_f}°F</span>
        </div>
        <div className={styles.metric}>
          <span className={styles.label}>Humidity</span>
          <span className={styles.value}>{room.sensors.humidity_rh}%</span>
        </div>
        <div className={styles.metric}>
          <span className={styles.label}>VPD</span>
          <span className={styles.value}>{room.sensors.vpd_kpa.toFixed(2)}</span>
        </div>
      </div>

      <div className={styles.footer}>
        <span className={styles.updated}>
          Updated {Math.round((Date.now() - room.lastUpdated) / 1000)}s ago
        </span>
        <span className={styles.clickHint}>Click for details →</span>
      </div>
    </div>
  )
}
