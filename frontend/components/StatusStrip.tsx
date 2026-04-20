'use client'

import React from 'react'

interface StatusStripProps {
  systemName?: string
  state: string
  confidence: 'high' | 'moderate' | 'low'
  timeToImpactLabel: string
}

export function StatusStrip({
  systemName = 'System',
  state,
  confidence,
  timeToImpactLabel,
}: StatusStripProps) {
  const stateColors: Record<string, string> = {
    Stable: '#38BDF8',
    'Drift forming': '#FBBF24',
    'Instability forming': '#FB923C',
    Critical: '#EF4444',
  }

  const color = stateColors[state] || '#38BDF8'
  const confidenceLabel = confidence === 'high' ? 'High' : confidence === 'moderate' ? 'Moderate' : 'Low'

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        height: '40px',
        background: 'rgba(10, 14, 26, 0.95)',
        borderBottom: `1px solid rgba(${color === '#38BDF8' ? '56, 189, 248' : color === '#FBBF24' ? '251, 191, 36' : color === '#FB923C' ? '251, 146, 60' : '239, 68, 68'}, 0.2)`,
        display: 'flex',
        alignItems: 'center',
        paddingLeft: '24px',
        paddingRight: '24px',
        gap: '32px',
        fontFamily: 'system-ui, -apple-system, sans-serif',
        fontSize: '13px',
        color: '#e2e8f0',
        letterSpacing: '0.05em',
        zIndex: 100,
      }}
    >
      {/* System Name */}
      <div style={{ fontWeight: 600, minWidth: '120px' }}>{systemName}</div>

      {/* State */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <div
          style={{
            width: '8px',
            height: '8px',
            borderRadius: '50%',
            background: color,
            boxShadow: `0 0 8px ${color}`,
          }}
        />
        <span style={{ fontWeight: 500 }}>{state}</span>
      </div>

      {/* Confidence */}
      <div style={{ color: '#94a3b8', fontSize: '12px' }}>
        Confidence: <span style={{ color }}>{confidenceLabel}</span>
      </div>

      {/* Time to Impact */}
      {timeToImpactLabel && (
        <div style={{ marginLeft: 'auto', color, fontWeight: 600 }}>
          {timeToImpactLabel}
        </div>
      )}
    </div>
  )
}
