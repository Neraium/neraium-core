'use client'

import React, { useState, useEffect } from 'react'
import styles from './GrowOpDashboard.module.css'
import RoomCard from './RoomCard'
import RoomDetail from './RoomDetailLegacy'
import BuildingOverview from './BuildingOverview'

interface SensorReading {
  temperature_f: number
  humidity_rh: number
  co2_ppm: number
  vpd_kpa: number
  ph: number
  ec_ms: number
  ppfd_umol: number
  irrigation_ml: number
}

export interface Room {
  id: string
  name: string
  floor: number
  zone: string
  status: 'optimal' | 'warning' | 'critical'
  sensors: SensorReading
  capacity: number
  plantCount: number
  stage: string
  lastUpdated: number
}

interface BuildingData {
  name: string
  address: string
  totalCapacity: number
  rooms: Room[]
}

const generateMockRoomData = (): BuildingData => {
  const now = Date.now()

  const rooms: Room[] = [
    {
      id: 'room-1',
      name: 'Vegetative Zone A',
      floor: 1,
      zone: 'North',
      status: 'optimal',
      sensors: {
        temperature_f: 75.5,
        humidity_rh: 62.3,
        co2_ppm: 1350,
        vpd_kpa: 1.1,
        ph: 6.1,
        ec_ms: 1.6,
        ppfd_umol: 600,
        irrigation_ml: 250,
      },
      capacity: 1000,
      plantCount: 950,
      stage: 'Vegetative (Week 3-4)',
      lastUpdated: now,
    },
    {
      id: 'room-2',
      name: 'Vegetative Zone B',
      floor: 1,
      zone: 'South',
      status: 'optimal',
      sensors: {
        temperature_f: 74.8,
        humidity_rh: 63.1,
        co2_ppm: 1400,
        vpd_kpa: 1.05,
        ph: 6.05,
        ec_ms: 1.55,
        ppfd_umol: 620,
        irrigation_ml: 260,
      },
      capacity: 1000,
      plantCount: 920,
      stage: 'Vegetative (Week 2-3)',
      lastUpdated: now,
    },
    {
      id: 'room-3',
      name: 'Flowering Zone A',
      floor: 2,
      zone: 'North',
      status: 'optimal',
      sensors: {
        temperature_f: 72.2,
        humidity_rh: 55.4,
        co2_ppm: 1500,
        vpd_kpa: 1.3,
        ph: 6.2,
        ec_ms: 1.8,
        ppfd_umol: 750,
        irrigation_ml: 280,
      },
      capacity: 1200,
      plantCount: 1150,
      stage: 'Flowering (Week 4-5)',
      lastUpdated: now,
    },
    {
      id: 'room-4',
      name: 'Flowering Zone B',
      floor: 2,
      zone: 'South',
      status: 'critical',
      sensors: {
        temperature_f: 88.3,
        humidity_rh: 42.1,
        co2_ppm: 920,
        vpd_kpa: 2.4,
        ph: 6.7,
        ec_ms: 2.4,
        ppfd_umol: 650,
        irrigation_ml: 180,
      },
      capacity: 1200,
      plantCount: 1100,
      stage: 'Flowering (Week 6-7)',
      lastUpdated: now,
    },
    {
      id: 'room-5',
      name: 'Drying & Curing',
      floor: 3,
      zone: 'Central',
      status: 'optimal',
      sensors: {
        temperature_f: 65.0,
        humidity_rh: 48.5,
        co2_ppm: 400,
        vpd_kpa: 0.8,
        ph: 0,
        ec_ms: 0,
        ppfd_umol: 0,
        irrigation_ml: 0,
      },
      capacity: 5000,
      plantCount: 0,
      stage: 'Drying & Curing',
      lastUpdated: now,
    },
  ]

  return {
    name: 'Neraium Grow Facility 01',
    address: '1234 Cultivation Ave, Growing City, GC 12345',
    totalCapacity: 5400,
    rooms,
  }
}

export default function GrowOpDashboard() {
  const [buildingData, setBuildingData] = useState<BuildingData | null>(null)
  const [selectedRoom, setSelectedRoom] = useState<Room | null>(null)
  const [view, setView] = useState<'overview' | 'detail'>('overview')

  useEffect(() => {
    const data = generateMockRoomData()
    setBuildingData(data)

    const interval = setInterval(() => {
      setBuildingData((prev) => {
        if (!prev) return prev

        return {
          ...prev,
          rooms: prev.rooms.map((room) => {
            if (room.id !== 'room-4') {
              return room
            }

            const randomFactor = Math.sin(Date.now() / 5000) * 0.5 + 0.5
            const degradedTemp = 85 + randomFactor * 4
            const degradedHumidity = 38 + randomFactor * 6
            const degradedVpd = 2.1 + randomFactor * 0.5

            return {
              ...room,
              sensors: {
                ...room.sensors,
                temperature_f: parseFloat(degradedTemp.toFixed(1)),
                humidity_rh: parseFloat(degradedHumidity.toFixed(1)),
                vpd_kpa: parseFloat(degradedVpd.toFixed(2)),
              },
              status:
                degradedTemp > 87 || degradedVpd > 2.2 ? 'critical' : 'warning',
              lastUpdated: Date.now(),
            }
          }),
        }
      })
    }, 3000)

    return () => clearInterval(interval)
  }, [])

  if (!buildingData) {
    return <div className={styles.loading}>Loading facility data...</div>
  }

  const handleRoomClick = (room: Room) => {
    setSelectedRoom(room)
    setView('detail')
  }

  const handleBackToOverview = () => {
    setView('overview')
    setSelectedRoom(null)
  }

  return (
    <div className={styles.container}>
      {view === 'overview' ? (
        <>
          <div className={styles.header}>
            <div>
              <h1>{buildingData.name}</h1>
              <p className={styles.address}>{buildingData.address}</p>
            </div>
            <div className={styles.stats}>
              <div className={styles.stat}>
                <span className={styles.label}>Total Plants</span>
                <span className={styles.value}>
                  {buildingData.rooms.reduce((sum, r) => sum + r.plantCount, 0)}
                </span>
              </div>
              <div className={styles.stat}>
                <span className={styles.label}>Capacity Usage</span>
                <span className={styles.value}>
                  {Math.round(
                    (buildingData.rooms.reduce((sum, r) => sum + r.plantCount, 0) /
                      buildingData.totalCapacity) *
                      100
                  )}
                  %
                </span>
              </div>
              <div className={styles.stat}>
                <span className={styles.label}>Active Rooms</span>
                <span className={styles.value}>{buildingData.rooms.length}</span>
              </div>
            </div>
          </div>

          <BuildingOverview building={buildingData} />

          <div className={styles.roomsGrid}>
            <h2>Room Status</h2>
            <div className={styles.grid}>
              {buildingData.rooms.map((room) => (
                <RoomCard
                  key={room.id}
                  room={room}
                  onClick={() => handleRoomClick(room)}
                />
              ))}
            </div>
          </div>
        </>
      ) : (
        <RoomDetail
          room={selectedRoom!}
          onBack={handleBackToOverview}
          building={buildingData}
        />
      )}
    </div>
  )
}
