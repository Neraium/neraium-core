'use client'

import React from 'react'
import { motion } from 'framer-motion'

interface SubsystemData {
  interpolatedDrift: number
  interpolatedStability: number
  interpolatedCoherence: number
  interpolatedConfidence: number
}

interface SubsystemSignalsProps {
  data: SubsystemData
}

export function SubsystemSignals({ data }: SubsystemSignalsProps) {
  // Generate signal descriptions based on current values
  const signals = [
    {
      name: 'Airflow',
      status: data.interpolatedDrift > 0.5 ? '↑ influence (destabilizing)' : '→ stable',
      direction: data.interpolatedDrift > 0.5 ? 'unstable' : 'stable',
    },
    {
      name: 'Structural Coupling',
      status: data.interpolatedCoherence > 0.6 ? '✓ coherent' : '⚠ weakening',
      direction: data.interpolatedCoherence > 0.6 ? 'stable' : 'unstable',
    },
    {
      name: 'System Stability',
      status: data.interpolatedStability > 0.6 ? '✓ maintained' : '↓ declining',
      direction: data.interpolatedStability > 0.6 ? 'stable' : 'unstable',
    },
  ]

  const getSignalColor = (direction: string) => {
    if (direction === 'stable') return '#38BDF8'
    if (direction === 'unstable') return '#FB923C'
    return '#94a3b8'
  }

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
        gap: '24px',
        maxWidth: '100%',
        padding: '0 40px',
      }}
    >
      {signals.map((signal, index) => (
        <motion.div
          key={signal.name}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.1, duration: 0.5 }}
          style={{
            display: 'flex',
            flexDirection: 'column',
            gap: '6px',
          }}
        >
          <div
            style={{
              fontSize: '11px',
              letterSpacing: '0.08em',
              color: '#94a3b8',
              textTransform: 'uppercase',
              fontWeight: 700,
            }}
          >
            {signal.name}
          </div>
          <div
            style={{
              fontSize: '13px',
              color: getSignalColor(signal.direction),
              fontWeight: 500,
            }}
          >
            {signal.status}
          </div>
        </motion.div>
      ))}
    </div>
  )
}
