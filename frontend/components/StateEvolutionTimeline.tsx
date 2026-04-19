'use client'

import React from 'react'
import { motion } from 'framer-motion'

interface TimelinePoint {
  label: string
  timestamp: string
  status: 'baseline' | 'forming' | 'advanced' | 'critical'
  isCurrent: boolean
}

interface StateEvolutionTimelineProps {
  points: TimelinePoint[]
}

const statusColors = {
  baseline: { bg: 'rgba(126, 159, 46, 0.1)', border: '#7E9F2E', dot: '#7E9F2E' },
  forming: { bg: 'rgba(216, 163, 93, 0.1)', border: '#D8A35D', dot: '#D8A35D' },
  advanced: { bg: 'rgba(249, 115, 22, 0.1)', border: '#F97316', dot: '#F97316' },
  critical: { bg: 'rgba(201, 76, 76, 0.1)', border: '#C94C4C', dot: '#C94C4C' },
}

export const StateEvolutionTimeline: React.FC<StateEvolutionTimelineProps> = ({
  points,
}) => {
  return (
    <div className="px-6 py-4 bg-louddpax-surface rounded border border-louddpax-border">
      <p className="text-xs text-louddpax-muted tracking-widest uppercase mb-4">
        State Evolution
      </p>

      <div className="relative">
        {/* Timeline line */}
        <div className="absolute left-4 top-0 bottom-0 w-px bg-gradient-to-b from-louddpax-primary/20 via-louddpax-primary/10 to-transparent" />

        {/* Timeline points */}
        <div className="space-y-3 pl-12">
          {points.map((point, idx) => {
            const colors = statusColors[point.status]
            return (
              <motion.div
                key={`${point.label}-${idx}`}
                className={`px-3 py-2 rounded border relative ${
                  point.isCurrent ? 'border-louddpax-primary' : ''
                }`}
                style={{
                  backgroundColor: colors.bg,
                  borderColor: colors.border,
                }}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: idx * 0.1, duration: 0.3 }}
              >
                {/* Timeline dot */}
                <motion.div
                  className="absolute -left-7 top-1/2 -translate-y-1/2 w-2 h-2 rounded-full"
                  style={{ backgroundColor: colors.dot }}
                  animate={
                    point.isCurrent
                      ? {
                          boxShadow: [
                            `0 0 0px ${colors.dot}`,
                            `0 0 8px ${colors.dot}`,
                            `0 0 0px ${colors.dot}`,
                          ],
                        }
                      : {}
                  }
                  transition={{
                    duration: 1.5,
                    repeat: Infinity,
                    ease: 'easeInOut',
                  }}
                />

                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-xs text-louddpax-muted tracking-widest uppercase">
                      {point.label}
                    </p>
                    <p className="text-sm text-louddpax-text font-mono text-xs">
                      {point.timestamp}
                    </p>
                  </div>
                  {point.isCurrent && (
                    <div className="text-xs text-louddpax-primary font-bold tracking-widest">
                      NOW
                    </div>
                  )}
                </div>
              </motion.div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
