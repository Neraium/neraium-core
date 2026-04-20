'use client'

import React from 'react'
import { motion } from 'framer-motion'
import { LiveSystem } from '@/lib/simulation'
import {
  growStageFromSystem,
  growStageLabel,
  growStageColor,
  urgencyFromStage,
  timeToImpactText,
} from '@/lib/growMapping'

interface Props {
  system: LiveSystem
}

const C = {
  text: '#e8e9eb',
  secondary: '#8a8f9a',
  muted: '#4a4f5a',
  neon: '#7cb342',
  orange: '#ff9e6d',
  red: '#e05c5c',
  surface: '#080a0d',
  border: 'rgba(255,255,255,0.04)',
}

export function GrowRiskIndicator({ system }: Props) {
  const stage = growStageFromSystem(system.structuralDrift, system.severity)
  const color = growStageColor(stage)
  const label = growStageLabel(stage)
  const urgency = urgencyFromStage(stage, system.trajectory)
  const timeText = timeToImpactText(system.structuralDrift, stage, system.onsetMinutes)

  const isCritical = stage === 'high_risk'
  const isStress = stage === 'stress_forming'

  return (
    <div style={{
      position: 'absolute',
      top: 18,
      left: '50%',
      transform: 'translateX(-50%)',
      zIndex: 20,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      gap: 6,
    }}>
      <motion.div
        animate={isCritical ? {
          boxShadow: [
            `0 0 0px ${color}00`,
            `0 0 20px ${color}30`,
            `0 0 0px ${color}00`,
          ],
        } : {}}
        transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          padding: '10px 18px',
          borderRadius: 8,
          background: `${color}10`,
          border: `1px solid ${color}30`,
          backdropFilter: 'blur(8px)',
        }}
      >
        {/* Status dot */}
        <motion.div
          animate={isCritical || isStress ? {
            scale: [1, 1.2, 1],
          } : {}}
          transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
          style={{
            width: 10,
            height: 10,
            borderRadius: '50%',
            background: color,
            boxShadow: `0 0 12px ${color}50`,
          }}
        />

        <div style={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <span style={{
              fontSize: 14,
              fontWeight: 900,
              color,
              letterSpacing: '0.02em',
            }}>
              {label}
            </span>
            <span style={{
              fontSize: 9,
              color: C.secondary,
              fontWeight: 600,
              letterSpacing: '0.04em',
            }}>
              {urgency}
            </span>
          </div>
          <span style={{
            fontSize: 9,
            color: C.secondary,
            fontWeight: 500,
            letterSpacing: '0.02em',
          }}>
            {timeText}
          </span>
        </div>
      </motion.div>

      {/* Operator moment — single trigger message */}
      {isCritical && (
        <motion.div
          initial={{ opacity: 0, y: -4 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          style={{
            padding: '6px 14px',
            borderRadius: 4,
            background: `${C.red}15`,
            border: `1px solid ${C.red}30`,
          }}
        >
          <span style={{
            fontSize: 9,
            color: C.red,
            fontWeight: 800,
            letterSpacing: '0.06em',
            textTransform: 'uppercase',
          }}>
            System entering instability window
          </span>
        </motion.div>
      )}
    </div>
  )
}
