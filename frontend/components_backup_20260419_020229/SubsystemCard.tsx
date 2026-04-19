'use client'

import React from 'react'
import { motion } from 'framer-motion'

interface SubsystemCardProps {
  name: string
  keyValue: number
  unit: string
  driftContribution: number
  trend: 'up' | 'down' | 'stable'
  sparklineData: number[]
  status: 'optimal' | 'warning' | 'critical'
}

const statusColors = {
  optimal: {
    text: '#7E9F2E',
    bg: 'rgba(126, 159, 46, 0.1)',
    border: 'rgba(126, 159, 46, 0.3)',
    spark: '#7E9F2E',
  },
  warning: {
    text: '#D8A35D',
    bg: 'rgba(216, 163, 93, 0.1)',
    border: 'rgba(216, 163, 93, 0.3)',
    spark: '#D8A35D',
  },
  critical: {
    text: '#C94C4C',
    bg: 'rgba(201, 76, 76, 0.1)',
    border: 'rgba(201, 76, 76, 0.3)',
    spark: '#C94C4C',
  },
}

const MiniSparkline: React.FC<{
  data: number[]
  color: string
}> = ({ data, color }) => {
  const width = 60
  const height = 30
  const padding = 4

  if (data.length === 0) return null

  const max = Math.max(...data)
  const min = Math.min(...data)
  const range = max - min || 1

  const points = data
    .map((value, idx) => {
      const x =
        padding + (idx / (data.length - 1 || 1)) * (width - padding * 2)
      const y =
        height -
        padding -
        ((value - min) / range) * (height - padding * 2)
      return `${x},${y}`
    })
    .join(' ')

  return (
    <svg width={width} height={height} className="overflow-visible">
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        opacity="0.6"
      />
      <polyline
        points={points}
        fill={color}
        opacity="0.1"
        stroke="none"
      />
    </svg>
  )
}

export const SubsystemCard: React.FC<SubsystemCardProps> = ({
  name,
  keyValue,
  unit,
  driftContribution,
  trend,
  sparklineData,
  status,
}) => {
  const colors = statusColors[status]
  const trendIndicators = {
    up: '↑',
    down: '↓',
    stable: '→',
  }

  return (
    <motion.div
      className="p-4 rounded border backdrop-blur-sm"
      style={{
        backgroundColor: colors.bg,
        borderColor: colors.border,
      }}
      whileHover={{ borderColor: colors.text, scale: 1.02 } ?? 0}
      transition={{ duration: 0.2 }}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div>
          <p className="text-xs text-louddpax-muted tracking-widest uppercase mb-1">
            {name}
          </p>
          <p className="text-2xl font-light tracking-tight" style={{ color: colors.text }}>
            {keyValue}
            <span className="text-xs ml-1 text-louddpax-muted">{unit}</span>
          </p>
        </div>
        <div
          className="text-lg font-light"
          style={{ color: colors.text }}
        >
          {trendIndicators[trend]}
        </div>
      </div>

      {/* Sparkline */}
      <div className="mb-4">
        <MiniSparkline data={sparklineData} color={colors.spark ?? 0} />
      </div>

      {/* Drift Contribution */}
      <div>
        <p className="text-xs text-louddpax-muted tracking-widest uppercase mb-2">
          Drift Contribution
        </p>
        <div className="flex items-center justify-between mb-1">
          <span className="text-sm" style={{ color: colors.text }}>
            {Math.round(driftContribution)}%
          </span>
        </div>
        <div className="h-1.5 bg-louddpax-border/30 rounded-full overflow-hidden">
          <motion.div
            className="h-full"
            style={{ backgroundColor: colors.text }}
            initial={{ width: 0 }}
            animate={{ width: `${driftContribution}%` }}
            transition={{ duration: 0.6, ease: 'easeOut' }}
          />
        </div>
      </div>
    </motion.div>
  )
}
