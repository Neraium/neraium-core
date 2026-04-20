'use client'

import React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { DecisionCommitment, getMomentumOpacity, getCommitmentGlowIntensity, isRecommendationLocked } from '@/lib/decisionCommitment'

interface ConsequenceIndicatorProps {
  timeToImpactLabel: string
  operatorFocus: string | null
  escalationLevel: number
  hasThresholdCrossed: boolean
  confidence?: 'high' | 'moderate' | 'low'
  actionOutcome?: string | null
  noActionConsequence?: string | null
  commitment?: DecisionCommitment | null
}

export function ConsequenceIndicator({
  timeToImpactLabel,
  operatorFocus,
  escalationLevel,
  hasThresholdCrossed,
  confidence,
  actionOutcome,
  noActionConsequence,
  commitment,
}: ConsequenceIndicatorProps) {
  // Only show when instability+ (escalationLevel >= 2)
  const shouldShow = escalationLevel >= 2

  // Color intensity based on escalation
  const colors = {
    2: { text: '#FBBF24', glow: 'rgba(251, 191, 36, 0.3)' }, // amber
    3: { text: '#EF4444', glow: 'rgba(239, 68, 68, 0.4)' }, // red
  }

  const color = colors[escalationLevel as 2 | 3] || colors[2]

  // Commitment-driven visual strength
  const commitmentOpacity = commitment ? getMomentumOpacity(commitment.momentum) : 1
  const glowIntensity = commitment ? getCommitmentGlowIntensity(commitment.commitmentScore) : 1
  const isLocked = commitment ? isRecommendationLocked(commitment.momentum, commitment.commitmentScore) : false

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

          {/* Operator Focus Line - Commitment-driven strength */}
          {operatorFocus && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{
                opacity: commitmentOpacity,
                textShadow: isLocked
                  ? `0 0 ${8 * glowIntensity}px ${color.text}${Math.round(100 * glowIntensity * 0.5).toString(16)}`
                  : `0 0 ${4 * glowIntensity}px ${color.text}${Math.round(100 * glowIntensity * 0.3).toString(16)}`
              }}
              transition={{
                delay: 0.2,
                duration: 0.5,
                opacity: { duration: 0.8 },
                textShadow: { duration: isLocked ? 0.1 : 0.6 }
              }}
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
                <span style={{ color: '#94a3b8', fontSize: '10px', marginLeft: '8px', opacity: 0.7 + commitmentOpacity * 0.2 }}>
                  [{confidence}]
                </span>
              )}
            </motion.div>
          )}

          {/* Action Outcome Projection - commitment-driven opacity */}
          {actionOutcome && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: commitmentOpacity * 0.85 }}
              transition={{ delay: 0.35, duration: 0.5, opacity: { duration: 0.8 } }}
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

          {/* No-Action Consequence - commitment-driven opacity */}
          {noActionConsequence && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: commitmentOpacity * 0.7 }}
              transition={{ delay: 0.45, duration: 0.5, opacity: { duration: 0.8 } }}
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
