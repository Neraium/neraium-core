'use client'

import React from 'react'
import { motion } from 'framer-motion'

type Trend = 'up' | 'down' | 'stable'

export interface SubsystemCardProps {
  name: string
  value: string
  status?: string
  trend?: Trend
  contribution?: number
  confidence?: number
  color?: string
}

function SubsystemCard({
  name,
  value,
  status = 'Stable',
  trend = 'stable',
  contribution = 0,
  confidence = 0,
  color = '#7E9F2E',
}: SubsystemCardProps) {
  const safeContribution =
    typeof contribution === 'number' && !Number.isNaN(contribution)
      ? Math.max(0, Math.min(100, contribution))
      : 0

  const safeConfidence =
    typeof confidence === 'number' && !Number.isNaN(confidence)
      ? Math.max(0, Math.min(100, confidence))
      : 0

  const trendLabel =
    trend === 'up' ? 'Rising' : trend === 'down' ? 'Falling' : 'Stable'

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className="rounded-2xl border p-4 backdrop-blur-sm"
      style={{
        backgroundColor: 'rgba(17,20,23,0.88)',
        borderColor: 'rgba(255,255,255,0.08)',
      }}
    >
      <div className="mb-3 flex items-start justify-between">
        <div>
          <div className="text-xs uppercase tracking-[0.18em] text-white/55">
            {name}
          </div>
          <div className="mt-2 text-2xl font-semibold text-white">{value}</div>
        </div>

        <div
          className="h-3 w-3 rounded-full"
          style={{
            backgroundColor: color,
            boxShadow: `0 0 12px ${color}`,
          }}
        />
      </div>

      <div className="mb-3 flex items-center justify-between text-sm">
        <span className="text-white/60">Status</span>
        <span className="text-white">{status}</span>
      </div>

      <div className="mb-3 flex items-center justify-between text-sm">
        <span className="text-white/60">Trend</span>
        <span className="text-white">{trendLabel}</span>
      </div>

      <div className="mb-2 text-xs uppercase tracking-[0.16em] text-white/45">
        Drift contribution
      </div>
      <div className="mb-4 h-2 w-full overflow-hidden rounded-full bg-white/10">
        <div
          className="h-full rounded-full"
          style={{
            width: `${safeContribution}%`,
            backgroundColor: color,
          }}
        />
      </div>

      <div className="mb-2 text-xs uppercase tracking-[0.16em] text-white/45">
        Confidence
      </div>
      <div className="h-2 w-full overflow-hidden rounded-full bg-white/10">
        <div
          className="h-full rounded-full bg-white/80"
          style={{ width: `${safeConfidence}%` }}
        />
      </div>
    </motion.div>
  )
}

export { SubsystemCard }

export default SubsystemCard
