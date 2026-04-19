'use client'

import React from 'react'
import { motion } from 'framer-motion'

interface Contributor {
  name: string
  percentage: number
  status: 'critical' | 'warning' | 'optimal'
}

interface IntelligenceRailProps {
  stateLabel: string
  driftScore: number
  confidence: number
  driftOnsetTime?: string
  topContributors: Contributor[]
  explanation: string
}

const C = {
  text: '#e8e9eb',
  secondary: '#8a8f9a',
  muted: '#4a4f5a',
  neon: '#7cb342',
  blue: '#5c9dff',
  orange: '#ff9e6d',
  red: '#e05c5c',
  surface: '#080a0d',
  surfaceHi: '#0f1116',
  border: 'rgba(255,255,255,0.04)',
}

export const IntelligenceRail: React.FC<IntelligenceRailProps> = ({
  stateLabel,
  driftScore,
  confidence,
  driftOnsetTime,
  topContributors,
  explanation,
}) => {
  const getStateStatusType = (): 'critical' | 'warning' | 'optimal' => {
    if (driftScore > 0.7) return 'critical'
    if (driftScore > 0.4) return 'warning'
    return 'optimal'
  }

  const statusType = getStateStatusType()
  const statusColor = statusType === 'critical' ? C.red : statusType === 'warning' ? C.orange : C.neon

  return (
    <div style={{
      width: '100%',
      background: C.surface,
      borderLeft: `1px solid ${C.border}`,
      padding: '20px 22px',
      display: 'flex',
      flexDirection: 'column',
      gap: 14,
      overflowY: 'auto',
      height: '100%',
    }}>
      {/* State Label */}
      <motion.div
        animate={{
          boxShadow:
            statusType === 'critical'
              ? [
                  '0 0 0px rgba(224, 92, 92, 0)',
                  '0 0 16px rgba(224, 92, 92, 0.15)',
                  '0 0 0px rgba(224, 92, 92, 0)',
                ]
              : 'none',
        }}
        transition={{ duration: 1.8, repeat: Infinity, ease: 'easeInOut' }}
        style={{
          padding: '14px 16px',
          borderRadius: 8,
          background: `${statusColor}08`,
          border: `1px solid ${statusColor}25`,
          borderLeft: `2px solid ${statusColor}`,
        }}
      >
        <p style={{ fontSize: 8, color: C.muted, fontWeight: 700, letterSpacing: '0.14em', textTransform: 'uppercase', marginBottom: 6 }}>
          Current State
        </p>
        <p style={{ fontSize: 18, fontWeight: 800, color: statusColor, letterSpacing: '-0.01em' }}>
          {stateLabel}
        </p>
      </motion.div>

      {/* Drift Score Gauge */}
      <div>
        <p style={{ fontSize: 8, color: C.muted, fontWeight: 700, letterSpacing: '0.14em', textTransform: 'uppercase', marginBottom: 8 }}>
          Drift Accumulation
        </p>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
          <span style={{ fontSize: 13, color: C.text, fontWeight: 900, fontVariantNumeric: 'tabular-nums' }}>
            {Math.round(driftScore * 100)}%
          </span>
          <span style={{ fontSize: 8, color: C.secondary, fontWeight: 700, letterSpacing: '0.08em' }}>
            {driftScore < 0.3 ? 'BASELINE' : driftScore < 0.6 ? 'FORMING' : 'ADVANCED'}
          </span>
        </div>
        <div style={{ height: 3, background: 'rgba(255,255,255,0.05)', borderRadius: 2, overflow: 'hidden' }}>
          <motion.div
            style={{
              height: '100%',
              borderRadius: 2,
              background:
                driftScore < 0.4
                  ? 'linear-gradient(90deg, #7cb342, #9fd65c)'
                  : driftScore < 0.7
                    ? 'linear-gradient(90deg, #ff9e6d, #ffb88c)'
                    : 'linear-gradient(90deg, #e05c5c, #e87878)',
              boxShadow: `0 0 10px ${driftScore < 0.4 ? C.neon : driftScore < 0.7 ? C.orange : C.red}25`,
            }}
            initial={{ width: 0 }}
            animate={{ width: `${driftScore * 100}%` }}
            transition={{ duration: 0.8, ease: 'easeOut' }}
          />
        </div>
      </div>

      {/* Confidence Score */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        padding: '10px 12px',
        background: C.surfaceHi,
        borderRadius: 6,
        border: `1px solid ${C.border}`,
      }}>
        <span style={{ fontSize: 8, color: C.muted, fontWeight: 700, letterSpacing: '0.14em', textTransform: 'uppercase' }}>
          Prediction Confidence
        </span>
        <span style={{ fontSize: 18, fontWeight: 900, color: C.text, fontVariantNumeric: 'tabular-nums' }}>
          {Math.round(confidence * 100)}
          <span style={{ fontSize: 10, marginLeft: 2, color: C.secondary }}>%</span>
        </span>
      </div>

      {/* Drift Onset Time */}
      {driftOnsetTime && (
        <div style={{
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          padding: '10px 12px',
          background: C.surfaceHi,
          borderRadius: 6,
          border: `1px solid ${C.border}`,
        }}>
          <span style={{ fontSize: 8, color: C.muted, fontWeight: 700, letterSpacing: '0.14em', textTransform: 'uppercase' }}>
            Drift Onset
          </span>
          <span style={{ fontSize: 11, color: C.text, fontWeight: 700, fontVariantNumeric: 'tabular-nums' }}>
            {driftOnsetTime}
          </span>
        </div>
      )}

      {/* Top Contributors */}
      <div>
        <p style={{ fontSize: 8, color: C.muted, fontWeight: 700, letterSpacing: '0.14em', textTransform: 'uppercase', marginBottom: 8 }}>
          Top Contributors
        </p>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {topContributors.slice(0, 3).map((contributor, idx) => {
            const ccol = contributor.status === 'critical' ? C.red : contributor.status === 'warning' ? C.orange : C.neon
            return (
              <motion.div
                key={`${contributor.name}-${idx}`}
                style={{
                  padding: '10px 12px',
                  borderRadius: 6,
                  background: `${ccol}08`,
                  border: `1px solid ${ccol}20`,
                  borderLeft: `2px solid ${ccol}`,
                }}
                animate={{ opacity: [0.75, 1, 0.75] }}
                transition={{ duration: 2.5, repeat: Infinity, ease: 'easeInOut', delay: idx * 0.25 }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
                  <span style={{ fontSize: 10, color: C.text, fontWeight: 700 }}>
                    {contributor.name}
                  </span>
                  <span style={{ fontSize: 10, color: ccol, fontWeight: 800 }}>
                    {Math.round(contributor.percentage)}%
                  </span>
                </div>
                <div style={{ height: 2, background: 'rgba(255,255,255,0.05)', borderRadius: 1, overflow: 'hidden' }}>
                  <motion.div
                    style={{
                      height: '100%',
                      background: ccol,
                      borderRadius: 1,
                      boxShadow: `0 0 6px ${ccol}30`,
                    }}
                    initial={{ width: 0 }}
                    animate={{ width: `${contributor.percentage}%` }}
                    transition={{ duration: 0.6, ease: 'easeOut' }}
                  />
                </div>
              </motion.div>
            )
          })}
        </div>
      </div>

      {/* System Analysis */}
      <div>
        <p style={{ fontSize: 8, color: C.muted, fontWeight: 700, letterSpacing: '0.14em', textTransform: 'uppercase', marginBottom: 6 }}>
          System Analysis
        </p>
        <p style={{ fontSize: 11, color: C.text, fontWeight: 500, lineHeight: 1.55 }}>
          {explanation}
        </p>
      </div>

      {/* Divider */}
      <div style={{ height: 1, background: C.border, marginTop: 'auto' }} />

      {/* Brand mark */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, paddingTop: 4 }}>
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" opacity={0.5}>
          <rect x="4" y="4" width="4" height="16" rx="1" fill={C.neon} opacity="0.9" />
          <rect x="10" y="4" width="4" height="16" rx="1" fill={C.neon} opacity="0.7" />
          <rect x="16" y="8" width="4" height="12" rx="1" fill={C.neon} opacity="0.5" />
        </svg>
        <p style={{ fontSize: 8, color: C.muted, fontWeight: 700, letterSpacing: '0.14em', textTransform: 'uppercase' }}>
          Neraium Intelligence
        </p>
      </div>
    </div>
  )
}
