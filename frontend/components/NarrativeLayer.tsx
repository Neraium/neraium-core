'use client'

import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

interface NarrativeLayerProps {
  text: string
  phase: string
  phaseProgress: number
}

export function NarrativeLayer({ text, phase, phaseProgress }: NarrativeLayerProps) {
  const [displayText, setDisplayText] = useState('')
  const [key, setKey] = useState(0)

  // Causal delay: 0.5-1s after phase shift (tied to stability)
  const stabilityDelay = phase === 'Stable' ? 500 : phase === 'Drift forming' ? 700 : 900

  useEffect(() => {
    // Reset on phase change
    if (phaseProgress < 0.1) {
      const timer = setTimeout(() => {
        setDisplayText(text)
        setKey(k => k + 1)
      }, stabilityDelay)
      return () => clearTimeout(timer)
    }
  }, [text, phaseProgress, stabilityDelay])

  // Fade timing tied to system stability
  // Stable: slower fade (longer visibility)
  // Critical: faster fade (more urgent)
  const fadeDuration = phase === 'Stable' ? 1.5 : phase === 'Critical' ? 0.6 : 1.0

  return (
    <AnimatePresence mode="wait">
      {displayText && (
        <motion.div
          key={key}
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 8 }}
          transition={{ duration: fadeDuration * 0.6, ease: 'easeInOut' }}
          style={{
            textAlign: 'center',
            maxWidth: '500px',
            margin: '0 auto',
          }}
        >
          <p
            style={{
              fontSize: '15px',
              lineHeight: '1.5',
              color: '#cbd5e1',
              fontWeight: 400,
              margin: 0,
              letterSpacing: '0.25px',
              // System voice: direct, minimal
              fontFamily: '"SF Mono", Monaco, Courier, monospace',
            }}
          >
            {displayText}
          </p>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
