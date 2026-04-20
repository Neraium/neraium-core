'use client'

import React from 'react'
import { motion } from 'framer-motion'

interface SubsystemIndicatorsProps {
  drift: number
  stability: number
  coherence: number
}

export function SubsystemIndicators({
  drift,
  stability,
  coherence,
}: SubsystemIndicatorsProps) {
  // Infer subsystem status from metrics
  const airflow = drift > 0.6 ? '↑' : drift > 0.3 ? '→' : '↓'
  const climate = stability < 0.4 ? '↓' : stability < 0.7 ? '→' : '↑'
  const irrigation = coherence < 0.4 ? '↓' : coherence < 0.7 ? '→' : '↑'
  const plantStress = drift + (1 - stability) > 1.2 ? '↑' : '↓'

  const subsystems = [
    { label: 'Airflow', indicator: airflow },
    { label: 'Climate', indicator: climate },
    { label: 'Irrigation', indicator: irrigation },
    { label: 'Plant stress', indicator: plantStress },
  ]

  return (
    <motion.div
      initial={{ opacity: 1, x: 0 }}
      animate={{ opacity: 1, x: 0 }}
      style={{
        position: 'fixed',
        left: '24px',
        top: '100px',
        display: 'flex',
        flexDirection: 'column',
        gap: '12px',
        fontSize: '12px',
        color: '#94a3b8',
        fontFamily: 'system-ui, -apple-system, sans-serif',
        letterSpacing: '0.05em',
        zIndex: 40,
      }}
    >
      {subsystems.map((subsystem) => (
        <div key={subsystem.label} style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <span style={{ fontWeight: 500, minWidth: '80px' }}>{subsystem.label}</span>
          <span
            style={{
              fontSize: '14px',
              color: subsystem.indicator === '↑' ? '#EF4444' : subsystem.indicator === '↓' ? '#38BDF8' : '#94a3b8',
              fontWeight: 600,
            }}
          >
            {subsystem.indicator}
          </span>
        </div>
      ))}
    </motion.div>
  )
}
