'use client'

import React, { useState, useEffect } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { useLiveSimulation } from '@/lib/simulation'
import { FacilityOverview } from './FacilityOverview'
import { RoomDetailView } from './RoomDetailView'

export function LouddpaxDashboard() {
  const {
    live,
    isPlaying,
    setIsPlaying,
    selectedRoomId,
    setSelectedRoomId,
    stepScenario,
  } = useLiveSimulation()

  const [view, setView] = useState<'facility' | 'room'>('facility')
  const [booted, setBooted] = useState(false)

  // Boot sequence — gives the "system coming online" feel
  useEffect(() => {
    const t = setTimeout(() => setBooted(true), 400)
    return () => clearTimeout(t)
  }, [])

  const handleSelectRoom = (id: string) => {
    setSelectedRoomId(id)
    setView('room')
  }

  const handleBack = () => {
    setView('facility')
    setSelectedRoomId(null)
  }

  return (
    <div style={{
      width: '100vw', height: '100vh',
      background: '#030405',
      overflow: 'hidden',
      position: 'relative',
    }}>
      {/* Subtle scanline texture */}
      <div style={{
        position: 'absolute', inset: 0,
        background: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.15) 2px, rgba(0,0,0,0.15) 4px)',
        pointerEvents: 'none',
        zIndex: 100,
        opacity: 0.3,
      }} />

      <AnimatePresence mode="popLayout">
        {view === 'facility' ? (
          <motion.div
            key="facility"
            style={{ width: '100%', height: '100%' }}
            initial={{ opacity: 0, scale: 0.98, filter: 'blur(4px)' }}
            animate={{ opacity: booted ? 1 : 0, scale: booted ? 1 : 0.98, filter: booted ? 'blur(0px)' : 'blur(4px)' }}
            exit={{ opacity: 0, scale: 1.01, filter: 'blur(4px)' }}
            transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
          >
            <FacilityOverview
              system={live}
              onSelectRoom={handleSelectRoom}
              isPlaying={isPlaying}
              onTogglePlay={() => setIsPlaying(p => !p)}
              onStep={stepScenario}
            />
          </motion.div>
        ) : (
          <motion.div
            key="room"
            style={{ width: '100%', height: '100%' }}
            initial={{ opacity: 0, scale: 1.01, filter: 'blur(4px)' }}
            animate={{ opacity: 1, scale: 1, filter: 'blur(0px)' }}
            exit={{ opacity: 0, scale: 0.98, filter: 'blur(4px)' }}
            transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
          >
            <RoomDetailView
              system={live}
              roomId={selectedRoomId!}
              onBack={handleBack}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
