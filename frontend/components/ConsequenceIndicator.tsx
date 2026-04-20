'use client'

import React from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ActionDecisionResult } from '@/lib/actionEvaluation'

interface ConsequenceIndicatorProps {
  timeToImpactLabel: string
  operatorFocus: string | null
  escalationLevel: number
  hasThresholdCrossed: boolean
  confidence?: 'high' | 'moderate' | 'low'
  actionOutcome?: string | null
  noActionConsequence?: string | null
  actionDecision?: ActionDecisionResult | null
}

export function ConsequenceIndicator({
  timeToImpactLabel,
  operatorFocus,
  escalationLevel,
  hasThresholdCrossed,
  confidence,
  actionOutcome,
  noActionConsequence,
  actionDecision,
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

          {/* Primary Action with Alternatives (New Multi-Action View) */}
          {actionDecision ? (
            <>
              {/* Primary Action */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.2, duration: 0.5 }}
                style={{
                  fontSize: '12px',
                  color: color.text,
                  fontWeight: 600,
                  letterSpacing: '0.4px',
                  textAlign: 'center',
                  maxWidth: '340px',
                }}
              >
                Primary: {actionDecision.primaryAction.label}
                {confidence && (
                  <span style={{ color: '#94a3b8', fontSize: '10px', marginLeft: '8px' }}>
                    [{confidence}]
                  </span>
                )}
              </motion.div>

              {/* Primary Outcome */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.3, duration: 0.5 }}
                style={{
                  fontSize: '11px',
                  color: '#cbd5e1',
                  fontWeight: 400,
                  letterSpacing: '0.3px',
                  textAlign: 'center',
                  marginTop: '4px',
                  maxWidth: '340px',
                }}
              >
                ↳ {actionDecision.primaryAction.outcome.primary}
              </motion.div>

              {/* Primary Continuation if present */}
              {actionDecision.primaryAction.outcome.continuation && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.35, duration: 0.5 }}
                  style={{
                    fontSize: '10px',
                    color: '#94a3b8',
                    fontWeight: 400,
                    letterSpacing: '0.3px',
                    textAlign: 'center',
                    marginTop: '2px',
                    maxWidth: '340px',
                  }}
                >
                  → {actionDecision.primaryAction.outcome.continuation}
                </motion.div>
              )}

              {/* Alternative Actions */}
              {actionDecision.alternatives.length > 0 && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.42, duration: 0.5 }}
                  style={{
                    fontSize: '10px',
                    color: '#94a3b8',
                    fontWeight: 400,
                    letterSpacing: '0.3px',
                    textAlign: 'center',
                    marginTop: '8px',
                    maxWidth: '340px',
                  }}
                >
                  {actionDecision.alternatives.map((alt, idx) => (
                    <div key={idx} style={{ marginTop: idx === 0 ? 0 : '3px' }}>
                      Alternative: {alt.label}
                      <div style={{ fontSize: '9px', marginTop: '1px' }}>
                        ↳ {alt.outcome.primary}
                      </div>
                    </div>
                  ))}
                </motion.div>
              )}

              {/* No-Action Consequence */}
              {actionDecision.noActionConsequence && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.5, duration: 0.5 }}
                  style={{
                    fontSize: '10px',
                    color: '#94a3b8',
                    fontWeight: 400,
                    letterSpacing: '0.3px',
                    textAlign: 'center',
                    marginTop: '6px',
                    maxWidth: '340px',
                    fontStyle: 'italic',
                  }}
                >
                  if no action: {actionDecision.noActionConsequence}
                </motion.div>
              )}

              {/* Timing Sensitivity */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.57, duration: 0.5 }}
                style={{
                  fontSize: '10px',
                  color: '#cbd5e1',
                  fontWeight: 500,
                  letterSpacing: '0.2px',
                  textAlign: 'center',
                  marginTop: '4px',
                  maxWidth: '340px',
                }}
              >
                {actionDecision.primaryAction.timingSensitivity}
              </motion.div>

              {/* Recommendation Stability */}
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.64, duration: 0.5 }}
                style={{
                  fontSize: '9px',
                  color: '#64748b',
                  fontWeight: 400,
                  letterSpacing: '0.2px',
                  textAlign: 'center',
                  marginTop: '2px',
                  maxWidth: '340px',
                }}
              >
                Recommendation stability: {actionDecision.recommendationStability}
              </motion.div>
            </>
          ) : (
            <>
              {/* Fallback to old format if no action decision */}
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
            </>
          )}
        </motion.div>
      )}
    </AnimatePresence>
  )
}
