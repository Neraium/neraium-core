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

const statusColors = {
  optimal: 'text-louddpax-primary',
  warning: 'text-louddpax-drift',
  critical: 'text-louddpax-critical',
}

const statusBgColors = {
  optimal: 'bg-louddpax-primary/10',
  warning: 'bg-louddpax-drift/10',
  critical: 'bg-louddpax-critical/10',
}

const statusBorderColors = {
  optimal: 'border-louddpax-primary/30',
  warning: 'border-louddpax-drift/30',
  critical: 'border-louddpax-critical/30',
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

  return (
    <div className="w-80 bg-louddpax-surface border-l border-louddpax-border p-6 flex flex-col gap-6 overflow-y-auto" suppressHydrationWarning>
      {/* State Label */}
      <motion.div
        className={`px-4 py-3 rounded border ${statusBgColors[statusType]} ${statusBorderColors[statusType]}`}
        animate={{
          boxShadow:
            statusType === 'critical'
              ? [
                  '0 0 0px rgba(201, 76, 76, 0)',
                  '0 0 12px rgba(201, 76, 76, 0.2)',
                  '0 0 0px rgba(201, 76, 76, 0)',
                ]
              : 'none',
        }}
        transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
      >
        <p className="text-xs text-louddpax-muted tracking-widest uppercase mb-2">
          Current State
        </p>
        <p
          className={`text-xl font-light tracking-tight ${statusColors[statusType]}`}
        >
          {stateLabel}
        </p>
      </motion.div>

      {/* Divider */}
      <div className="h-px bg-louddpax-border/50" />

      {/* Drift Score Gauge */}
      <div>
        <p className="text-xs text-louddpax-muted tracking-widest uppercase mb-3">
          Drift Accumulation
        </p>
        <div className="space-y-2">
          <div className="flex items-center justify-between mb-2">
            <span suppressHydrationWarning className="text-sm text-louddpax-text">
              {Math.round(driftScore * 100)}%
            </span>
            <span className="text-xs text-louddpax-muted">
              {driftScore < 0.3 ? 'BASELINE' : driftScore < 0.6 ? 'FORMING' : 'ADVANCED'}
            </span>
          </div>
          <div className="h-2 bg-louddpax-border rounded-full overflow-hidden">
            <motion.div
              className="h-full rounded-full"
              style={{
                background:
                  driftScore < 0.4
                    ? 'linear-gradient(90deg, #7E9F2E, #A3E635)'
                    : driftScore < 0.7
                      ? 'linear-gradient(90deg, #D8A35D, #E8B878)'
                      : 'linear-gradient(90deg, #C94C4C, #E87878)',
              }}
              initial={{ width: 0 }}
              animate={{ width: `${driftScore * 100}%` }}
              transition={{ duration: 0.8, ease: 'easeOut' }}
            />
          </div>
        </div>
      </div>

      {/* Confidence Score */}
      <div>
        <p className="text-xs text-louddpax-muted tracking-widest uppercase mb-2">
          Prediction Confidence
        </p>
        <p className="text-2xl font-light text-louddpax-text">
          {Math.round(confidence * 100)}
          <span className="text-sm ml-1 text-louddpax-muted">%</span>
        </p>
      </div>

      {/* Drift Onset Time */}
      {driftOnsetTime && (
        <>
          <div className="h-px bg-louddpax-border/50" />
          <div>
            <p className="text-xs text-louddpax-muted tracking-widest uppercase mb-2">
              Drift Onset
            </p>
            <p className="text-sm text-louddpax-text font-mono">
              {driftOnsetTime}
            </p>
          </div>
        </>
      )}

      {/* Top Contributors */}
      <div>
        <p className="text-xs text-louddpax-muted tracking-widest uppercase mb-3">
          Top Contributors
        </p>
        <div className="space-y-2">
          {topContributors.slice(0, 3).map((contributor, idx) => (
            <motion.div
              key={`${contributor.name}-${idx}`}
              className={`px-3 py-2 rounded border ${statusBgColors[contributor.status]} ${statusBorderColors[contributor.status]}`}
              animate={{
                opacity: [0.7, 1, 0.7],
              }}
              transition={{
                duration: 2,
                repeat: Infinity,
                ease: 'easeInOut',
                delay: idx * 0.2,
              }}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-louddpax-text font-medium">
                  {contributor.name}
                </span>
                <span
                  className={`text-xs font-bold ${statusColors[contributor.status]}`}
                >
                  {Math.round(contributor.percentage)}%
                </span>
              </div>
              <div className="h-1 bg-louddpax-border/30 rounded-full overflow-hidden">
                <motion.div
                  className="h-full"
                  style={{
                    background:
                      contributor.status === 'critical'
                        ? '#C94C4C'
                        : contributor.status === 'warning'
                          ? '#D8A35D'
                          : '#7E9F2E',
                  }}
                  initial={{ width: 0 }}
                  animate={{ width: `${contributor.percentage}%` }}
                  transition={{ duration: 0.6, ease: 'easeOut' }}
                />
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* Plain English Explanation */}
      <div>
        <p className="text-xs text-louddpax-muted tracking-widest uppercase mb-2">
          System Analysis
        </p>
        <p className="text-sm text-louddpax-text leading-relaxed">
          {explanation}
        </p>
      </div>

      {/* Divider */}
      <div className="h-px bg-louddpax-border/50" />

      {/* Powered by Neraium */}
      <div className="flex flex-col items-center gap-2 pt-2">
        <svg width="24" height="24" viewBox="0 0 24 24" className="opacity-60">
          <rect x="4" y="4" width="4" height="16" fill="#7E9F2E" rx="0.5" />
          <rect x="10" y="4" width="4" height="16" fill="#7E9F2E" rx="0.5" />
          <rect x="16" y="8" width="4" height="12" fill="#7E9F2E" rx="0.5" />
        </svg>
        <p className="text-xs text-louddpax-muted tracking-widest">
          Powered by Neraium
        </p>
      </div>
    </div>
  )
}
