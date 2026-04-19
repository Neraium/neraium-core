'use client'

import React from 'react'
import styles from './BuildingOverview.module.css'
import type { Room } from './GrowOpDashboard'

interface Props {
  building: {
    name: string
    rooms: Room[]
  }
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

export default function BuildingOverview({ building }: Props) {
  const floorGroups = building.rooms.reduce(
    (acc, room) => {
      if (!acc[room.floor]) {
        acc[room.floor] = []
      }
      acc[room.floor].push(room)
      return acc
    },
    {} as Record<number, Room[]>
  )

  const sortedFloors = Object.keys(floorGroups)
    .map(Number)
    .sort((a, b) => b - a)

  const criticalCount = building.rooms.filter(
    (r) => r.status === 'critical'
  ).length
  const warningCount = building.rooms.filter(
    (r) => r.status === 'warning'
  ).length

  return (
    <div className={styles.overview}>
      <div className={styles.alerts}>
        {criticalCount > 0 && (
          <div className={styles.alert + ' ' + styles.critical}>
            ⚠️ {criticalCount} room{criticalCount > 1 ? 's' : ''} in critical
            condition
          </div>
        )}
        {warningCount > 0 && (
          <div className={styles.alert + ' ' + styles.warning}>
            ℹ️ {warningCount} room{warningCount > 1 ? 's' : ''} with warnings
          </div>
        )}
        {criticalCount === 0 && warningCount === 0 && (
          <div className={styles.alert + ' ' + styles.optimal}>
            ✓ All systems nominal
          </div>
        )}
      </div>

      <div className={styles.buildingVisualization}>
        {sortedFloors.map((floor) => (
          <div key={floor} className={styles.floor}>
            <div className={styles.floorLabel}>Floor {floor}</div>
            <div className={styles.roomsRow}>
              {floorGroups[floor].map((room) => (
                <div
                  key={room.id}
                  className={styles.room}
                  style={{
                    borderColor: getStatusColor(room.status),
                    backgroundColor:
                      room.status === 'optimal'
                        ? 'rgba(16, 185, 129, 0.1)'
                        : room.status === 'warning'
                          ? 'rgba(245, 158, 11, 0.1)'
                          : 'rgba(239, 68, 68, 0.1)',
                  }}
                  title={`${room.name} - ${room.status}`}
                >
                  <div className={styles.roomInfo}>
                    <div className={styles.roomName}>{room.name}</div>
                    <div className={styles.roomMiniStats}>
                      <span>{room.plantCount} plants</span>
                      <span>{Math.round(room.sensors.temperature_f)}°</span>
                    </div>
                  </div>
                  <div
                    className={styles.statusIndicator}
                    style={{ backgroundColor: getStatusColor(room.status) }}
                  />
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
