'use client'

import React, { useMemo } from 'react'
import styles from './StateTimeline.module.css'
import type { Room } from './GrowOpDashboard'

interface Props {
  room: Room
  historyData: Record<string, number[]>
}

type RoomStateType = 'stable' | 'drifting' | 'critical'

interface StateEvent {
  time: string
  state: RoomStateType
  description: string
  reason: string
}

export default function StateTimeline({ room, historyData }: Props) {
  const events = useMemo<StateEvent[]>(() => {
    if (room.id !== 'room-4') {
      return [
        { time: '12h ago', state: 'stable', description: 'Baseline established', reason: 'All metrics nominal' },
        { time: 'now', state: 'stable', description: 'Stable', reason: 'Maintained conditions' },
      ]
    }

    // Room 4 is degrading - show state evolution
    const tempHistory = historyData['temperature_f'] || []
    const humidHistory = historyData['humidity_rh'] || []

    const degradationStart = Math.max(0, tempHistory.length - 60)
    const isCritical = tempHistory.length > 80

    return [
      {
        time: '12h ago',
        state: 'stable',
        description: 'Baseline established',
        reason: 'Room running nominal',
      },
      {
        time: degradationStart > 40 ? '2h ago' : 'monitoring',
        state: 'drifting',
        description: 'Drift detected in climate relationships',
        reason: 'Temperature rising while humidity falling',
      },
      ...(isCritical
        ? [
            {
              time: 'now',
              state: 'critical' as RoomStateType,
              description: 'Critical threshold breached',
              reason: 'VPD > 2.0 kPa, intervention window closing',
            },
          ]
        : [
            {
              time: 'now',
              state: 'drifting' as RoomStateType,
              description: 'Accelerating drift',
              reason: 'System moving away from optimal geometry',
            },
          ]),
    ]
  }, [room.id, historyData])

  const getStateColor = (state: RoomStateType) => {
    switch (state) {
      case 'stable':
        return '#22c55e'
      case 'drifting':
        return '#f59e0b'
      case 'critical':
        return '#ef4444'
    }
  }

  return (
    <div className={styles.container}>
      <h2>Room State Evolution</h2>
      <div className={styles.timeline}>
        {events.map((event, idx) => (
          <div key={idx} className={styles.event}>
            <div className={styles.timelineMarker}>
              <div
                className={styles.dot}
                style={{ backgroundColor: getStateColor(event.state) }}
              />
              {idx < events.length - 1 && (
                <div
                  className={styles.line}
                  style={{
                    background: `linear-gradient(180deg, ${getStateColor(event.state)} 0%, ${getStateColor(events[idx + 1].state)} 100%)`,
                  }}
                />
              )}
            </div>
            <div className={styles.content}>
              <div className={styles.time}>{event.time}</div>
              <div className={styles.state} style={{ color: getStateColor(event.state) }}>
                {event.state.toUpperCase()}
              </div>
              <div className={styles.description}>{event.description}</div>
              <div className={styles.reason}>{event.reason}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
