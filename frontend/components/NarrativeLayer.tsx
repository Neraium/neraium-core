'use client'

import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

interface NarrativeLayerProps {
  text: string
  phase: string
  phaseProgress: number
  diagnostics?: {
    origin?: string | null
    propagationPath?: string | null
    whyNow?: string | null
    currentRiskZone?: string | null
  }
}

export function NarrativeLayer({
  text,
  phase,
  phaseProgress,
  diagnostics,
}: NarrativeLayerProps) {
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

  // Construct diagnostic narrative (minimal, system voice)
  let diagnosticLine = ''
  if (phase !== 'Stable' && diagnostics) {
    const parts: string[] = []

    // Add origin if we have it
    if (diagnostics.origin && phase !== 'Drift forming') {
      parts.push(diagnostics.origin)
    }

    // Add why now if we have it
    if (diagnostics.whyNow) {
      parts.push(diagnostics.whyNow)
    }

    // Add current risk zone if in critical phase
    if ((phase === 'Instability forming' || phase === 'Critical') && diagnostics.currentRiskZone) {
      parts.push(`Risk: ${diagnostics.currentRiskZone}`)
    }

    diagnosticLine = parts.join(' • ')
  }

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
            maxWidth: '600px',
            margin: '0 auto',
          }}
        >
          {/* Primary escalation narrative */}
          <p
            style={{
              fontSize: '15px',
              lineHeight: '1.5',
              color: '#cbd5e1',
              fontWeight: 400,
              margin: 0,
              letterSpacing: '0.25px',
              fontFamily: '"SF Mono", Monaco, Courier, monospace',
            }}
          >
            {displayText}
          </p>

          {/* Diagnostic context (origin, why now, risk) */}
          {diagnosticLine && (
            <p
              style={{
                fontSize: '12px',
                lineHeight: '1.4',
                color: '#94a3b8',
                fontWeight: 400,
                margin: '8px 0 0 0',
                letterSpacing: '0.2px',
                fontFamily: '"SF Mono", Monaco, Courier, monospace',
              }}
            >
              {diagnosticLine}
            </p>
          )}
        </motion.div>
      )}
    </AnimatePresence>
  )
}
