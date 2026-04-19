'use client'

import React from 'react'
import { motion } from 'framer-motion'

interface TopBarProps {
  facilityName: string
  roomName: string
  growthPhase: string
  isLive?: boolean
  globalHealthScore?: number
}

export const TopBar: React.FC<TopBarProps> = ({
  facilityName,
  roomName,
  growthPhase,
  isLive = true,
  globalHealthScore = 92,
}) => {
  return (
    <div className="bg-louddpax-bg border-b border-louddpax-border px-8 py-6">
      <div className="max-w-[1920px] mx-auto flex items-center justify-between gap-8">
        {/* Left: Louddpax branding and location */}
        <div className="flex items-start gap-6">
          <div className="flex items-start gap-4">
            {/* Louddpax Logo */}
            <svg width="48" height="48" viewBox="0 0 48 48" className="flex-shrink-0">
              <circle cx="24" cy="24" r="22" fill="none" stroke="#7E9F2E" strokeWidth="1.5" opacity="0.3"/>
              <circle cx="24" cy="24" r="16" fill="none" stroke="#7E9F2E" strokeWidth="1" opacity="0.2"/>
              <path d="M 24 10 Q 32 14 32 24 Q 32 34 24 38 Q 16 34 16 24 Q 16 14 24 10" fill="#7E9F2E" opacity="0.8"/>
              <circle cx="24" cy="24" r="4" fill="#050607"/>
            </svg>
          </div>
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="text-xs font-bold tracking-wider text-louddpax-primary uppercase">
                Louddpax
              </span>
              {isLive && (
                <motion.div
                  className="flex items-center gap-1.5"
                  animate={{ opacity: [0.5, 1, 0.5] }}
                  transition={{ duration: 1.5, repeat: Infinity }}
                >
                  <div className="w-1.5 h-1.5 rounded-full bg-louddpax-primary" />
                  <span className="text-xs text-louddpax-primary tracking-widest">
                    LIVE
                  </span>
                </motion.div>
              )}
            </div>
            <h1 className="text-4xl font-light tracking-tight text-louddpax-text">
              {facilityName}
            </h1>
            <p className="text-xs text-louddpax-muted tracking-widest uppercase mt-2">
              {roomName}
            </p>
          </div>
        </div>

        {/* Center: Growth phase */}
        <div className="flex-1 flex justify-center">
          <div className="px-6 py-3 bg-louddpax-surface rounded border border-louddpax-border">
            <p className="text-xs text-louddpax-muted tracking-widest uppercase mb-1">
              Growth Phase
            </p>
            <p className="text-lg font-light text-louddpax-text tracking-tight">
              {growthPhase}
            </p>
          </div>
        </div>

        {/* Right: Global health chip */}
        <div className="flex items-center gap-8">
          <motion.div
            className="px-6 py-4 bg-gradient-to-br from-louddpax-primary/10 to-louddpax-primary/5 rounded border border-louddpax-primary/30"
            animate={{
              boxShadow: [
                '0 0 0px rgba(126, 159, 46, 0)',
                '0 0 12px rgba(126, 159, 46, 0.2)',
                '0 0 0px rgba(126, 159, 46, 0)',
              ],
            }}
            transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
          >
            <p className="text-xs text-louddpax-muted tracking-widest uppercase mb-2">
              System Health
            </p>
            <div className="flex items-baseline gap-2">
              <span className="text-3xl font-light tracking-tight text-louddpax-primary">
                {globalHealthScore}
              </span>
              <span className="text-xs text-louddpax-muted uppercase">%</span>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  )
}
