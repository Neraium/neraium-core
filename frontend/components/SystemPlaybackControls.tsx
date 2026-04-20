'use client'

import React from 'react'
import { motion } from 'framer-motion'

interface PlaybackControlsProps {
  isPlaying: boolean
  speed: number
  onTogglePlay: () => void
  onSpeedChange: (speed: number) => void
}

export function PlaybackControls({
  isPlaying,
  speed,
  onTogglePlay,
  onSpeedChange,
}: PlaybackControlsProps) {
  const speeds = [0.5, 1, 1.5, 2]

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.5, duration: 0.6 }}
      style={{
        display: 'flex',
        gap: '20px',
        alignItems: 'center',
        backgroundColor: 'rgba(2, 6, 23, 0.7)',
        border: '1px solid rgba(148, 163, 184, 0.2)',
        borderRadius: '8px',
        padding: '12px 16px',
        backdropFilter: 'blur(8px)',
      }}
    >
      {/* Play/Pause Button */}
      <motion.button
        onClick={onTogglePlay}
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.95 }}
        style={{
          background: 'none',
          border: 'none',
          color: '#38BDF8',
          cursor: 'pointer',
          fontSize: '20px',
          width: '32px',
          height: '32px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          borderRadius: '4px',
          transition: 'background-color 0.2s ease',
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.backgroundColor = 'rgba(56, 189, 248, 0.1)'
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.backgroundColor = 'transparent'
        }}
      >
        {isPlaying ? '⏸' : '▶'}
      </motion.button>

      {/* Speed Controls */}
      <div
        style={{
          display: 'flex',
          gap: '8px',
          alignItems: 'center',
          borderLeft: '1px solid rgba(148, 163, 184, 0.2)',
          paddingLeft: '16px',
        }}
      >
        <span
          style={{
            fontSize: '11px',
            color: '#94a3b8',
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
            fontWeight: 600,
          }}
        >
          Speed
        </span>
        {speeds.map((s) => (
          <motion.button
            key={s}
            onClick={() => onSpeedChange(s)}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            style={{
              background: speed === s ? 'rgba(56, 189, 248, 0.2)' : 'transparent',
              border: speed === s ? '1px solid rgba(56, 189, 248, 0.5)' : '1px solid rgba(148, 163, 184, 0.2)',
              color: speed === s ? '#38BDF8' : '#94a3b8',
              cursor: 'pointer',
              padding: '6px 10px',
              borderRadius: '4px',
              fontSize: '12px',
              fontWeight: 600,
              transition: 'all 0.2s ease',
            }}
          >
            {s}x
          </motion.button>
        ))}
      </div>
    </motion.div>
  )
}
