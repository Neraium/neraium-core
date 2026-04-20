'use client'

import React from 'react'
import { motion, AnimatePresence } from 'framer-motion'

interface ConsequenceIndicatorProps {
  timeToImpactLabel: string
  operatorFocus: string | null
  escalationLevel: number
  hasThresholdCrossed: boolean
  confidence?: 'high' | 'moderate' | 'low'
  actionOutcome?: string | null
  noActionConsequence?: string | null
}

export function ConsequenceIndicator({
  timeToImpactLabel,
  operatorFocus,
  escalationLevel,
  hasThresholdCrossed,
  confidence,
  actionOutcome,
  noActionConsequence,
}: ConsequenceIndicatorProps) {
  // Only show when instability+ (escalationLevel >= 2)
  const shouldShow = escalationLevel >= 2

  // Color intensity based on escalation
  const colors = {
    2: { text: '#FBBF24', glow: 'rgba(251, 191, 36, 0.3)' }, // amber
    3: { text: '#EF4444', glow: 'rgba(239, 68, 68, 0.4)' }, // red
  }

  const color = colors[escalationLevel as 2 | 3] || colors[2]

  return (
    <AnimatePresence>
      {shouldShow && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.4 }}
          style={{
            position: 'fixed',
            top: '60px',
            left: '50%',
            transform: 'translateX(-50%)',
            zIndex: 50,
            pointerEvents: 'none',
          }}
        >
          {/* Time-to-Impact Indicator */}
          {timeToImpactLabel && (
            <motion.div
              animate={{
                textShadow: hasThresholdCrossed
                  ? [`0 0 0px ${color.text}`, `0 0 12px ${color.text}`, `0 0 0px ${color.text}`]
                  : `0 0 8px ${color.glow}`,
              }}
              transition={{
                duration: hasThresholdCrossed ? 0.3 : 2,
                repeat: hasThresholdCrossed ? 1 : Infinity,
              }}
              style={{
                fontSize: '13px',
                color: color.text,
                fontWeight: 600,
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
                textAlign: 'center',
                marginBottom: '8px',
              }}
            >
              {timeToImpactLabel}
            </motion.div>
          )}

          {/* Operator Focus Line */}
          {operatorFocus && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2, duration: 0.5 }}
              style={{
                fontSize: '12px',
                color: color.text,
                fontWeight: 500,
                letterSpacing: '0.4px',
                textAlign: 'center',
                maxWidth: '340px',
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
              }}
            >
              {operatorFocus}
              {confidence && (
                <span style={{ color: '#94a3b8', fontSize: '10px', marginLeft: '8px' }}>
                  [{confidence}]
                </span>
              )}
            </motion.div>
          )}

          {/* Action Outcome Projection */}
          {actionOutcome && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.35, duration: 0.5 }}
              style={{
                fontSize: '11px',
                color: '#cbd5e1',
                fontWeight: 400,
                letterSpacing: '0.3px',
                textAlign: 'center',
                marginTop: '6px',
                maxWidth: '340px',
              }}
            >
              ↳ {actionOutcome}
            </motion.div>
          )}

          {/* No-Action Consequence */}
          {noActionConsequence && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.45, duration: 0.5 }}
              style={{
                fontSize: '11px',
                color: '#94a3b8',
                fontWeight: 400,
                letterSpacing: '0.3px',
                textAlign: 'center',
                marginTop: '4px',
                maxWidth: '340px',
                fontStyle: 'italic',
              }}
            >
              if no action: {noActionConsequence}
            </motion.div>
          )}
        </motion.div>
      )}
    </AnimatePresence>
  )
}
