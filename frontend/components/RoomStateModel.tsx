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
    <g key={name}>
      <circle
        cx={x}
        cy={y}
        r="24"
        fill={statusColors[status].bg}
        opacity="0.15"
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
        fill={statusColors[status].bg}
        opacity="0.3"
        animate={{ opacity: [0.3, 0.5, 0.3] }}
        transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
        suppressHydrationWarning
      />
      <circle cx={x} cy={y} r="12" fill={statusColors[status].bg} suppressHydrationWarning />

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
    </g>
  )
}

const ConnectionLine: React.FC<{
  x1: number
  y1: number
  x2: number
  y2: number
  tension: number
  driftScore: number
}> = ({ x1, y1, x2, y2, tension, driftScore }) => {
  const distance = Math.max(0, Math.hypot(x2 - x1, y2 - y1))
  const midX = (x1 + x2) / 2
  const midY = (y1 + y2) / 2
  const safeStrokeWidth = Math.max(0.5, 1.5 + tension * 1.5)

  const isCritical = driftScore > 0.7
  const glowColor = isCritical
    ? 'rgba(201, 76, 76, 0.4)'
    : 'rgba(126, 159, 46, 0.3)'

  return (
    <g>
      <defs>
        <filter id="lineGlow">
          <feGaussianBlur stdDeviation="2" result="coloredBlur" />
          <feMerge>
            <feMergeNode in="coloredBlur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      <motion.line
        x1={x1}
        y1={y1}
        x2={x2}
        y2={y2}
        stroke={isCritical ? '#C94C4C' : '#7E9F2E'}
        strokeWidth={safeStrokeWidth}
        opacity={Math.max(0.1, Math.min(1, 0.3 + driftScore * 0.4))}
        filter="url(#lineGlow)"
        animate={{
          opacity: isCritical
            ? [0.4, 0.7, 0.4]
            : [0.3, 0.5, 0.3],
        }}
        transition={{
          duration: isCritical ? 1.5 : 2.5,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
        suppressHydrationWarning
      />

      {/* Tension indicator glow */}
      {tension > 0.5 && (
        <motion.circle
          cx={midX}
          cy={midY}
          r={3 + tension * 2}
          fill="none"
          stroke={isCritical ? '#C94C4C' : '#D8A35D'}
          strokeWidth="0.5"
          opacity="0.5"
          animate={{ opacity: [0.3, 0.8, 0.3] }}
          transition={{ duration: 2, repeat: Infinity, ease: 'easeInOut' }}
          suppressHydrationWarning
        />
      )}
    </g>
  )
}

const DriftField: React.FC<{ driftScore: number }> = ({ driftScore }) => {
  return (
    <defs>
      <filter id="noiseFilter">
        <feTurbulence
          type="fractalNoise"
          baseFrequency="0.03"
          numOctaves="4"
          result="noise"
          seed="2"
        />
        <feDisplacementMap
          in="SourceGraphic"
          in2="noise"
          scale={Math.max(2, driftScore * 8)}
          xChannelSelector="R"
          yChannelSelector="G"
        />
      </filter>
    </defs>
  )
}

export const RoomStateModel: React.FC<RoomStateModelProps> = ({
  roomState,
  subsystems,
}) => {
  const subsystemAngles = [0, 60, 120, 180, 240, 300]
  const radius = 100

  const coreColors = {
    stable: { bg: '#7E9F2E', ring: 'rgba(126, 159, 46, 0.5)' },
    drift: { bg: '#D8A35D', ring: 'rgba(216, 163, 93, 0.5)' },
    instability: { bg: '#F97316', ring: 'rgba(249, 115, 22, 0.5)' },
    critical: { bg: '#C94C4C', ring: 'rgba(201, 76, 76, 0.5)' },
  }

  const tension = useMemo(
    () =>
      subsystems.reduce((sum, s) => {
        const multiplier =
          s.status === 'critical' ? 1.0 : s.status === 'warning' ? 0.6 : 0.2
        return sum + multiplier
      }, 0) / subsystems.length,
    [subsystems]
  )

  return (
    <div className="flex items-center justify-center w-full h-[500px] bg-louddpax-surface rounded-lg border border-louddpax-border overflow-hidden relative">
      <svg
        viewBox="0 0 400 400"
        className="w-full h-full"
        preserveAspectRatio="xMidYMid meet"
      >
        <DriftField driftScore={roomState.driftScore} />

        {/* Background subtle grid */}
        <defs>
          <pattern
            id="grid"
            width="40"
            height="40"
            patternUnits="userSpaceOnUse"
          >
            <path
              d="M 40 0 L 0 0 0 40"
              fill="none"
              stroke="rgba(126, 159, 46, 0.05)"
              strokeWidth="0.5"
            />
          </pattern>
        </defs>
        <rect
          width="400"
          height="400"
          fill="url(#grid)"
          opacity="0.3"
          filter="url(#noiseFilter)"
        />

        {/* Connection lines */}
        {subsystems.map((subsystem, idx) => {
          const angle = subsystemAngles[idx]
          const x = 200 + radius * Math.cos((angle * Math.PI) / 180)
          const y = 200 + radius * Math.sin((angle * Math.PI) / 180)
          const connectionTension =
            subsystem.status === 'critical'
              ? 1.0
              : subsystem.status === 'warning'
                ? 0.6
                : 0.2

          return (
            <ConnectionLine
              key={`line-${idx}`}
              x1={200}
              y1={200}
              x2={x}
              y2={y}
              tension={connectionTension}
              driftScore={roomState.driftScore}
            />
          )
        })}

        {/* Core node (room state) */}
        <g>
          <motion.circle
            cx="200"
            cy="200"
            r="48"
            fill={coreColors[roomState.status].bg}
            opacity="0.1"
            animate={{ opacity: [0.1, 0.15, 0.1] }}
            transition={{ duration: 4, repeat: Infinity, ease: 'easeInOut' }}
            suppressHydrationWarning
          />
          <circle
            cx="200"
            cy="200"
            r="44"
            fill="none"
            stroke={coreColors[roomState.status].bg}
            strokeWidth="2"
            opacity="0.3"
          />
          <motion.circle
            cx="200"
            cy="200"
            r="40"
            fill={coreColors[roomState.status].bg}
            opacity="0.2"
            animate={{ opacity: [0.2, 0.4, 0.2] }}
            transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
          />
          <circle cx="200" cy="200" r="36" fill={coreColors[roomState.status].bg} />

export default RoomStateModel
export { RoomStateModel }
