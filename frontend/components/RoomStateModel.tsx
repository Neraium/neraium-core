'use client'

import React from 'react'
import { motion } from 'framer-motion'

type AnyProps = {
  roomState?: {
    status?: string
    label?: string
  }
  className?: string
}

const statusColors: Record<string, { bg: string }> = {
  stable: { bg: '#22c55e' },
  drift: { bg: '#f59e0b' },
  critical: { bg: '#ef4444' },
}

function RoomStateModel({ roomState, className }: AnyProps) {
  const status = roomState?.status ?? 'stable'
  const label =
    roomState?.label ??
    (status === 'critical'
      ? 'Critical Divergence'
      : status === 'drift'
        ? 'Drift Detected'
        : 'Stable')

  const color = statusColors[status] ?? statusColors.stable

  const x = 200
  const y = 200

  return (
    <svg width="400" height="400" viewBox="0 0 400 400" className={className}>
      <motion.circle
        cx={x}
        cy={y}
        r={24}
        fill={color.bg}
        opacity={0.15}
        animate={{ r: [24, 28, 24] }}
        transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
      />

      <circle
        cx={x}
        cy={y}
        r={20}
        fill="none"
        stroke={color.bg}
        strokeWidth="2"
        opacity={0.4}
      />

      <motion.circle
        cx={x}
        cy={y}
        r={16}
        fill={color.bg}
        opacity={0.3}
        animate={{ opacity: [0.3, 0.6, 0.3] }}
        transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
      />

      <circle cx={x} cy={y} r={12} fill={color.bg} />

      <text
        x={x}
        y={y + 40}
        textAnchor="middle"
        fontSize="12"
        fill="#ffffff"
      >
        {label}
      </text>
    </svg>
  )
}


export default RoomStateModel
export { RoomStateModel }
