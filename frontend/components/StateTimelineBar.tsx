'use client'

import React from 'react'
import { DecisionUIState, DegradationStage, Severity } from '@/lib/decisionToUI'

interface Props {
  state: DecisionUIState
}

interface Segment {
  color: string
  bg: string
  pct: number
}

interface Marker {
  label: string
  pos: number
  color: string
}

const STAGE_ORDER = [
  DegradationStage.BASELINE,
  DegradationStage.EARLY_SHIFT,
  DegradationStage.EMERGING,
  DegradationStage.PERSISTENT,
  DegradationStage.ACCELERATED,
  DegradationStage.FAILURE_APPROACH,
]

function getSegments(stage: DegradationStage): Segment[] {
  const idx = STAGE_ORDER.indexOf(stage)
  if (idx === 0) {
    return [
      { color: '#7E9F2E', bg: 'rgba(126,159,46,0.15)', pct: 100 }
    ]
  }
  if (idx === 1) {
    return [
      { color: '#7E9F2E', bg: 'rgba(126,159,46,0.15)',  pct: 68 },
      { color: '#D8A35D', bg: 'rgba(216,163,93,0.15)', pct: 32 }
    ]
  }
  if (idx === 2 || idx === 3) {
    return [
      { color: '#7E9F2E', bg: 'rgba(126,159,46,0.15)',  pct: 52 },
      { color: '#D8A35D', bg: 'rgba(216,163,93,0.15)', pct: 26 },
      { color: '#C94C4C', bg: 'rgba(201,76,76,0.15)',   pct: 22 }
    ]
  }
  return [
    { color: '#7E9F2E', bg: 'rgba(126,159,46,0.15)',  pct: 35 },
    { color: '#D8A35D', bg: 'rgba(216,163,93,0.15)', pct: 20 },
    { color: '#C94C4C', bg: 'rgba(201,76,76,0.15)',   pct: 45 },
  ]
}

function getMarkers(stage: DegradationStage): Marker[] {
  const idx     = STAGE_ORDER.indexOf(stage)
  const markers: Marker[] = []

  if (idx >= 1) markers.push({ label: 'Drift detected',  pos: 66,  color: '#D8A35D' })
  if (idx >= 3) markers.push({ label: 'Regime change',   pos: 79,  color: '#C94C4C' })
  if (idx >= 4) markers.push({ label: 'Intervention',    pos: 90,  color: '#C94C4C' })

  const nowPos = [44, 80, 86, 90, 94, 98][idx] ?? 44
  markers.push({ label: 'Now', pos: nowPos, color: '#ffffff' })

  return markers
}

export function StateTimelineBar({ state }: Props) {
  const { statusHeader } = state
  const stage    = statusHeader.degradationStage
  const segments = getSegments(stage)
  const markers  = getMarkers(stage)

  return (
    <div style={{ padding: '10px 24px 14px' }}>
      {/* Header row */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        marginBottom: 8,
      }}>
        <span style={{
          fontSize: 10, color: '#374151', textTransform: 'uppercase',
          letterSpacing: '0.1em', fontWeight: 700,
        }}>
          State Timeline
        </span>
        <div style={{ display: 'flex', gap: 18 }}>
          {[
            { c: '#7E9F2E', l: 'Stable' },
            { c: '#D8A35D', l: 'Transition' },
            { c: '#C94C4C', l: 'Unstable' },
          ].map(({ c, l }) => (
            <span key={l} style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 10, color: '#4b5563' }}>
              <span style={{ width: 10, height: 3, borderRadius: 2, background: c, display: 'inline-block' }} />
              {l}
            </span>
          ))}
        </div>
      </div>

      {/* Bar + markers */}
      <div style={{ position: 'relative', paddingBottom: 22 }}>
        {/* Bar */}
        <div style={{
          height: 20, borderRadius: 3, overflow: 'hidden',
          display: 'flex', border: '1px solid rgba(255,255,255,0.05)',
        }}>
          {segments.map((seg, i) => (
            <div
              key={i}
              style={{
                width: `${seg.pct}%`,
                background: seg.bg,
                borderRight: i < segments.length - 1 ? '1px solid rgba(0,0,0,0.35)' : 'none',
                position: 'relative', overflow: 'hidden',
              }}
            >
              <div style={{
                position: 'absolute', top: 0, left: 0, right: 0, height: 2,
                background: seg.color, opacity: 0.7,
              }} />
            </div>
          ))}
        </div>

        {/* Markers */}
        {markers.map((mk, i) => {
          const isNow = mk.label === 'Now'
          return (
            <div
              key={i}
              style={{
                position: 'absolute', top: 0,
                left: `${mk.pos}%`,
                transform: 'translateX(-50%)',
                zIndex: isNow ? 20 : 10,
                pointerEvents: 'none',
              }}
            >
              {!isNow && (
                <div style={{
                  width: 0, height: 0,
                  borderLeft: '4px solid transparent',
                  borderRight: '4px solid transparent',
                  borderTop: `7px solid ${mk.color}`,
                  margin: '0 auto',
                }} />
              )}
              <div style={{
                width: isNow ? 2 : 1,
                height: isNow ? 20 : 16,
                background: mk.color,
                opacity: isNow ? 0.95 : 0.65,
                boxShadow: isNow ? `0 0 8px ${mk.color}88` : 'none',
                margin: '0 auto',
                marginTop: isNow ? 0 : 0,
              }} />
              {!isNow && (
                <div style={{
                  fontSize: 9, color: mk.color, opacity: 0.85,
                  whiteSpace: 'nowrap', textAlign: 'center',
                  marginTop: 3, letterSpacing: '0.04em', fontWeight: 600,
                }}>
                  {mk.label}
                </div>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
