'use client'

import React from 'react'
import styles from './RoomComparison.module.css'
import type { Room } from './GrowOpDashboard'

interface Props {
  currentRoom: Room
  allRooms: Room[]
}

export default function RoomComparison({ currentRoom, allRooms }: Props) {
  const similarRooms = allRooms.filter(
    r => r.id !== currentRoom.id && r.status === currentRoom.status
  )

  return (
    <div className={styles.container}>
      <h2>Room Comparison</h2>
      <div className={styles.grid}>
        {[currentRoom, ...similarRooms.slice(0, 2)].map(room => (
          <div key={room.id} className={styles.card}>
            <div className={styles.name}>{room.name}</div>
            <div className={`${styles.status} ${styles[room.status]}`}>
              {room.status}
            </div>
            <div className={styles.metrics}>
              <span>{room.sensors.temperature_f.toFixed(1)}°F</span>
              <span>{room.sensors.humidity_rh.toFixed(0)}%</span>
              <span>{room.sensors.vpd_kpa.toFixed(2)} VPD</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
