'use client'

import React from 'react'
import { motion } from 'framer-motion'
import { DecisionCommitment, getMomentumOpacity } from '@/lib/decisionCommitment'

interface CommittedActionOverlayProps {
  action: string | null
  actionOutcome?: string | null
  escalationLevel: number
  commitment?: DecisionCommitment | null
}

export function CommittedActionOverlay({
  action,
  actionOutcome,
  escalationLevel,
  commitment,
}: CommittedActionOverlayProps) {
  if (!action) return null

  const colors = {
    0: '#38BDF8',
    1: '#38BDF8',
    2: '#FBBF24',
    3: '#EF4444',
  }

  const color = colors[escalationLevel as keyof typeof colors] || colors[0]
  const opacity = commitment ? getMomentumOpacity(commitment.momentum) : 1

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity, y: 0 }}
      transition={{ duration: 0.6 }}
      style={{
        position: 'fixed',
        bottom: '140px',
        left: '50%',
        transform: 'translateX(-50%)',
        zIndex: 40,
        pointerEvents: 'none',
        textAlign: 'center',
      }}
    >
      {/* Action Label */}
      <motion.div
        style={{
          fontSize: '14px',
          fontWeight: 600,
          color: color,
          letterSpacing: '0.05em',
          marginBottom: '8px',
          textTransform: 'uppercase',
        }}
        animate={{
          textShadow: escalationLevel >= 2
            ? `0 0 8px ${color}`
            : 'none'
        }}
        transition={{ duration: 0.6 }}
      >
        {action}
      </motion.div>

      {/* Outcome Expectation */}
      {actionOutcome && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: Math.max(0.6, opacity) }}
          transition={{ delay: 0.2, duration: 0.6 }}
          style={{
            fontSize: '12px',
            color: '#cbd5e1',
            fontWeight: 400,
            letterSpacing: '0.3px',
          }}
        >
          ↳ {actionOutcome}
        </motion.div>
      )}
    </motion.div>
  )
}
