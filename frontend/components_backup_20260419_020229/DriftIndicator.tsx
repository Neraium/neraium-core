'use client'

import React from 'react'
import styles from './DriftIndicator.module.css'
import type { Room } from './GrowOpDashboard'

interface Props {
  driftScore: number
  room: Room
}

export default function DriftIndicator({ driftScore, room }: Props) {
  const getDriftLevel = (score: number) => {
    if (score < 20) return { label: 'Stable', color: '#22c55e' }
    if (score < 40) return { label: 'Minor Drift', color: '#84cc16' }
    if (score < 60) return { label: 'Moderate Drift', color: '#eab308' }
    if (score < 80) return { label: 'Serious Drift', color: '#f59e0b' }
    return { label: 'Critical Drift', color: '#ef4444' }
  }

  const level = getDriftLevel(room.id === 'room-4' ? driftScore : 5)

  return (
    <div className={styles.card}>
      <div className={styles.label}>System Drift</div>
      <div className={styles.gauge}>
        <div className={styles.gaugeBackground}>
          <div
            className={styles.gaugeFill}
            style={{
              width: `${Math.min(100, driftScore)}%`,
              backgroundColor: level.color,
            }}
          />
        </div>
      </div>
      <div className={styles.level} style={{ color: level.color }}>
        {level.label}
      </div>
      <div className={styles.subtext}>
        {room.id === 'room-4'
          ? driftScore > 70
            ? 'Room diverging from optimal path'
            : 'Relationships shifting between sensors'
          : 'System geometry stable'}
      </div>
    </div>
  )
}
