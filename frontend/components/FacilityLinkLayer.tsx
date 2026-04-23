'use client'

import React, { useMemo } from 'react'
import type { RoomStatus } from './TeslaAutopilotInterface'

function clamp01(x: number): number {
  if (!Number.isFinite(x)) return 0
  return Math.max(0, Math.min(1, x))
}

function statusColor(status: RoomStatus['status']): string {
  if (status === 'critical') return '#ef4444'
  if (status === 'warning') return '#f59e0b'
  return '#22c55e'
}

function stateWord(status: RoomStatus['status']): string {
  if (status === 'critical') return 'At risk'
  if (status === 'warning') return 'Transitioning'
  return 'Stable'
}

export function FacilityLinkLayer({
  rooms,
  primaryDriver,
  systemState,
}: {
  rooms: RoomStatus[]
  primaryDriver: string
  systemState: 'Stable' | 'Watch' | 'Alert'
}) {
  const model = useMemo(() => {
    const safeRooms = (rooms || []).filter(r => r && r.id)
    const affected = safeRooms
      .filter(r => r.status !== 'optimal' || clamp01(r.driftContribution) > 0.25)
      .sort((a, b) => clamp01(b.driftContribution) - clamp01(a.driftContribution))

    const origin = affected[0] || safeRooms.sort((a, b) => clamp01(b.driftContribution) - clamp01(a.driftContribution))[0]
    const path = affected.slice(0, 4)
    const stable = safeRooms.filter(r => !path.some(p => p.id === r.id))

    const driver = (primaryDriver || 'multi-mode').toString().replace(/-/g, ' ')
    const headline =
      path.length >= 3
        ? `Propagation across ${path.length} zones`
        : path.length === 2
          ? 'Coupled shift across 2 zones'
          : path.length === 1
            ? 'Localized shift in 1 zone'
            : 'Facility structure nominal'

    return { safeRooms, affected, origin, path, stable, driver, headline }
  }, [rooms, primaryDriver])

  const subtle = systemState === 'Stable'
  const originColor = statusColor(model.origin?.status || 'optimal')

  return (
    <div
      style={{
        background: 'rgba(5,6,7,0.46)',
        border: '1px solid rgba(255,255,255,0.06)',
        borderRadius: 14,
        padding: '14px 14px',
        display: 'grid',
        gap: 12,
        backdropFilter: 'blur(10px)',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 12 }}>
        <div style={{ minWidth: 0 }}>
          <div style={{ fontSize: 10, color: '#64748b', letterSpacing: '1px', textTransform: 'uppercase', fontWeight: 950 }}>
            Facility map
          </div>
          <div style={{ marginTop: 2, fontSize: 13, color: '#e2e8f0', fontWeight: 900, letterSpacing: 0.2 }}>
            {model.headline}
            <span style={{ color: '#94a3b8', fontWeight: 800 }}> · </span>
            <span style={{ color: '#cbd5e1', fontWeight: 800 }}>Driver:</span>{' '}
            <span style={{ color: '#e2e8f0', fontWeight: 850 }}>{model.driver}</span>
          </div>
        </div>

        <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexShrink: 0 }}>
          <div
            style={{
              fontSize: 10,
              letterSpacing: '1px',
              textTransform: 'uppercase',
              fontWeight: 950,
              color: subtle ? '#94a3b8' : '#e2e8f0',
              padding: '6px 10px',
              borderRadius: 999,
              background: subtle ? 'rgba(148,163,184,0.08)' : 'rgba(255,255,255,0.06)',
              border: '1px solid rgba(255,255,255,0.06)',
            }}
          >
            Zones: {model.safeRooms.length}
          </div>
          <div
            style={{
              fontSize: 10,
              letterSpacing: '1px',
              textTransform: 'uppercase',
              fontWeight: 950,
              color: subtle ? '#94a3b8' : '#e2e8f0',
              padding: '6px 10px',
              borderRadius: 999,
              background: subtle ? 'rgba(148,163,184,0.08)' : 'rgba(255,255,255,0.06)',
              border: '1px solid rgba(255,255,255,0.06)',
            }}
          >
            Stable: {model.stable.length}
          </div>
        </div>
      </div>

      {/* Propagation strip: origin -> coupled -> downstream */}
      <div
        style={{
          borderRadius: 12,
          background: 'rgba(255,255,255,0.028)',
          border: '1px solid rgba(255,255,255,0.055)',
          padding: '12px 12px',
          display: 'grid',
          gap: 10,
        }}
      >
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'baseline' }}>
          <div style={{ fontSize: 10, color: '#94a3b8', letterSpacing: '1px', textTransform: 'uppercase', fontWeight: 950 }}>
            Origin
          </div>
          <div style={{ fontSize: 12, color: '#e2e8f0', fontWeight: 900 }}>
            {model.origin?.shortName || model.origin?.id || '-'}
          </div>
          <div style={{ fontSize: 11, color: originColor, fontWeight: 900, letterSpacing: 0.5, textTransform: 'uppercase' }}>
            {stateWord(model.origin?.status || 'optimal')}
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
          {model.path.length > 0 ? (
            model.path.map((r, idx) => {
              const c = statusColor(r.status)
              const isOrigin = model.origin?.id && r.id === model.origin.id
              const pct = Math.round(clamp01(r.driftContribution) * 100)
              return (
                <React.Fragment key={r.id}>
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 8,
                      padding: '8px 10px',
                      borderRadius: 999,
                      background: isOrigin ? `${c}14` : 'rgba(255,255,255,0.04)',
                      border: `1px solid ${isOrigin ? c + '30' : 'rgba(255,255,255,0.06)'}`,
                      minWidth: 0,
                    }}
                  >
                    <span
                      style={{
                        width: 8,
                        height: 8,
                        borderRadius: 999,
                        background: c,
                        boxShadow: isOrigin ? `0 0 12px ${c}66` : 'none',
                        flexShrink: 0,
                      }}
                    />
                    <span style={{ fontSize: 12, color: '#e2e8f0', fontWeight: 850, whiteSpace: 'nowrap' }}>
                      {r.shortName || r.id}
                    </span>
                    <span style={{ fontSize: 11, color: c, fontWeight: 950, fontVariantNumeric: 'tabular-nums' }}>
                      {pct}%
                    </span>
                  </div>

                  {idx < model.path.length - 1 ? (
                    <div style={{ color: 'rgba(226,232,240,0.35)', fontWeight: 900, letterSpacing: 1 }}>
                      {'->'}
                    </div>
                  ) : null}
                </React.Fragment>
              )
            })
          ) : (
            <div style={{ fontSize: 12, color: '#94a3b8' }}>No propagation detected</div>
          )}
        </div>

        {model.stable.length > 0 ? (
          <details style={{ marginTop: 2 }}>
            <summary
              style={{
                cursor: 'pointer',
                userSelect: 'none',
                fontSize: 10,
                color: '#94a3b8',
                letterSpacing: '1px',
                textTransform: 'uppercase',
                fontWeight: 950,
              }}
            >
              +{model.stable.length} stable zones
            </summary>
            <div style={{ marginTop: 10, display: 'flex', gap: 10, flexWrap: 'wrap' }}>
              {model.stable.slice(0, 12).map(r => (
                <div
                  key={r.id}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: 7,
                    padding: '6px 8px',
                    borderRadius: 999,
                    background: 'rgba(255,255,255,0.03)',
                    border: '1px solid rgba(255,255,255,0.05)',
                    opacity: 0.8,
                  }}
                >
                  <span style={{ width: 6, height: 6, borderRadius: 999, background: statusColor(r.status), opacity: 0.55 }} />
                  <span style={{ fontSize: 11, color: '#cbd5e1', fontWeight: 750 }}>{r.shortName || r.id}</span>
                </div>
              ))}
              {model.stable.length > 12 ? (
                <div style={{ fontSize: 11, color: '#94a3b8', padding: '6px 0', fontWeight: 800 }}>
                  +{model.stable.length - 12} more
                </div>
              ) : null}
            </div>
          </details>
        ) : null}
      </div>
    </div>
  )
}

export default FacilityLinkLayer
