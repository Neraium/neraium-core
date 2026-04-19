'use client'

import React, { useEffect, useRef, useState } from 'react'
import { LiveSystem, LiveRoom } from '@/lib/simulation'

interface Props {
  system: LiveSystem
}

/* ─── Palette ─────────────────────────────────────────────────────────────── */

const C = {
  bg: '#030405',
  neon: '#7cb342',
  neonGlow: 'rgba(124,179,66,0.30)',
  blue: '#5c9dff',
  blueGlow: 'rgba(92,157,255,0.30)',
  orange: '#ff9e6d',
  orangeGlow: 'rgba(255,158,109,0.30)',
  red: '#e05c5c',
  redGlow: 'rgba(224,92,92,0.35)',
  text: '#e8e9eb',
  muted: '#3a3f4a',
}

/* ─── Layout ──────────────────────────────────────────────────────────────── */

const VB = 500
const CX = VB / 2
const CY = VB / 2

function healthColor(health: LiveRoom['status']) {
  return health === 'critical' ? C.red : health === 'warning' ? C.orange : C.neon
}

function healthGlow(health: LiveRoom['status']) {
  return health === 'critical' ? C.redGlow : health === 'warning' ? C.orangeGlow : C.neonGlow
}

/* ─── Component ───────────────────────────────────────────────────────────── */

export function SystemField({ system }: Props) {
  const { rooms, coherence, structuralDrift, stage, severity } = system
  const idx = ['baseline', 'early_shift', 'emerging', 'persistent', 'accelerated', 'failure_approach'].indexOf(stage)

  /* animation tick */
  const [tick, setTick] = useState(0)
  const lastRef = useRef(0)
  useEffect(() => {
    let raf: number
    const loop = (t: number) => {
      if (t - lastRef.current > 60) {
        lastRef.current = t
        setTick(x => x + 1)
      }
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [])

  const time = tick * 0.04

  /* node positions (5 rooms in pentagon) */
  const nodes = rooms.map((room, i) => {
    const angle = (i * 72 - 90) * (Math.PI / 180)
    const baseDist = 155
    const dist = baseDist + Math.sin(time + i * 1.2) * (idx === 0 ? 3 : idx <= 2 ? 6 : 10)

    /* drift deformation: pull critical nodes outward */
    const pull = room.driftContribution * 35
    const px = CX + Math.cos(angle) * (dist + pull)
    const py = CY + Math.sin(angle) * (dist + pull)

    /* critical jitter */
    const jx = idx >= 4 ? (Math.sin(time * 3 + i * 7) * 3) : 0
    const jy = idx >= 4 ? (Math.cos(time * 2.7 + i * 5) * 3) : 0

    return {
      ...room,
      x: px + jx,
      y: py + jy,
      angle,
    }
  })

  /* coherence ring segments */
  const segments = 48
  const ringRadius = 105
  const ringPath = Array.from({ length: segments }, (_, i) => {
    const a = (i / segments) * Math.PI * 2
    const aNext = ((i + 1) / segments) * Math.PI * 2
    const x1 = CX + Math.cos(a) * ringRadius
    const y1 = CY + Math.sin(a) * ringRadius
    const x2 = CX + Math.cos(aNext) * ringRadius
    const y2 = CY + Math.sin(aNext) * ringRadius
    return { x1, y1, x2, y2, i }
  })

  /* tension lines */
  const lines = nodes.map((node, i) => {
    const mx = (CX + node.x) / 2
    const my = (CY + node.y) / 2
    const perp = node.angle + Math.PI / 2
    const bend = (idx * 8 + structuralDrift * 25) * (node.driftContribution > 0.4 ? 1.5 : 0.5)
    const cpx = mx + Math.cos(perp) * bend + Math.sin(time + i) * 3
    const cpy = my + Math.sin(perp) * bend + Math.cos(time + i) * 3
    return {
      d: `M${CX.toFixed(1)},${CY.toFixed(1)} Q${cpx.toFixed(1)},${cpy.toFixed(1)} ${node.x.toFixed(1)},${node.y.toFixed(1)}`,
      col: healthColor(node.status),
      glow: healthGlow(node.status),
      broken: idx >= 4 && node.status === 'critical',
    }
  })

  /* drift contours */
  const contours = idx >= 1 ? [1, 2, 3].map(layer => ({
    rx: 180 + layer * 30 + Math.sin(time * 0.5 + layer) * 10,
    ry: 140 + layer * 25 + Math.cos(time * 0.4 + layer) * 8,
    rot: layer * 15 + Math.sin(time * 0.3) * 5,
    opacity: (structuralDrift * 0.15) / layer,
    col: idx >= 3 ? C.orange : C.blue,
  })) : []

  /* future path trace */
  const futureTrace = idx >= 1 ? (() => {
    const pts: Array<{ x: number; y: number }> = []
    for (let i = 0; i < 12; i++) {
      const t = i / 11
      const a = t * Math.PI * 1.5 - Math.PI / 2
      const r = 60 + t * 180 + structuralDrift * 50
      pts.push({
        x: CX + Math.cos(a + structuralDrift * 0.5) * r,
        y: CY + Math.sin(a + structuralDrift * 0.5) * r,
      })
    }
    return pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')
  })() : ''

  const coreColor =
    severity === 'HIGH' ? C.red
    : severity === 'ELEVATED' ? C.orange
    : severity === 'MODERATE' ? C.blue
    : C.neon

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative', overflow: 'hidden' }}>
      {/* ambient field gradient */}
      <div style={{
        position: 'absolute', inset: 0,
        background: `radial-gradient(ellipse at ${(CX / VB) * 100}% ${(CY / VB) * 100}%, ${coreColor}08 0%, transparent 60%)`,
        transition: 'background 1.2s ease',
      }} />

      <svg viewBox={`0 0 ${VB} ${VB}`} width="100%" height="100%" preserveAspectRatio="xMidYMid meet" style={{ display: 'block' }}>
        <defs>
          <filter id="sf-glow-neon" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="5" /><feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge></filter>
          <filter id="sf-glow-blue" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="5" /><feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge></filter>
          <filter id="sf-glow-orange" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="5" /><feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge></filter>
          <filter id="sf-glow-red" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="6" /><feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge></filter>
          <filter id="sf-core" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="10" /><feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge></filter>
        </defs>

        <style>{`
          @keyframes sfBreathe { 0%,100%{transform:scale(1)} 50%{transform:scale(1.05)} }
          @keyframes sfFlow { 0%{stroke-dashoffset:20} 100%{stroke-dashoffset:0} }
          .sf-node { transform-origin:center; animation: sfBreathe ${idx >= 4 ? '0.9s' : idx >= 2 ? '1.6s' : '3s'} ease-in-out infinite; }
          .sf-edge { stroke-dasharray: 3 3; animation: sfFlow ${idx >= 4 ? '0.6s' : '1.8s'} linear infinite; }
        `}</style>

        {/* drift contours */}
        {contours.map((c, i) => (
          <ellipse key={i} cx={CX} cy={CY} rx={c.rx} ry={c.ry}
            fill="none" stroke={c.col} strokeWidth={0.6} opacity={c.opacity}
            transform={`rotate(${c.rot.toFixed(1)} ${CX} ${CY})`}
          />
        ))}

        {/* future path trace */}
        {futureTrace && (
          <path d={futureTrace} fill="none" stroke={C.blue} strokeWidth={1} opacity={0.12} strokeDasharray="4 6" />
        )}

        {/* coherence ring */}
        {ringPath.map(seg => {
          const segCoherence = coherence - Math.abs(Math.sin(seg.i / segments * Math.PI * 2 + time * 0.3)) * (1 - coherence)
          const opacity = Math.max(0, segCoherence * 0.5)
          const col = segCoherence < 0.3 ? C.red : segCoherence < 0.6 ? C.orange : C.neon
          return (
            <line key={seg.i} x1={seg.x1} y1={seg.y1} x2={seg.x2} y2={seg.y2}
              stroke={col} strokeWidth={1.2} opacity={opacity}
            />
          )
        })}

        {/* tension lines */}
        {lines.map((line, i) => {
          const filter = line.col === C.red ? 'sf-glow-red' : line.col === C.orange ? 'sf-glow-orange' : line.col === C.blue ? 'sf-glow-blue' : 'sf-glow-neon'
          if (line.broken) {
            const dx = nodes[i].x - CX
            const dy = nodes[i].y - CY
            const len = Math.sqrt(dx * dx + dy * dy)
            const nx = dx / len
            const ny = dy / len
            const gap = 16
            const mid = len / 2
            return (
              <g key={i}>
                <line x1={CX} y1={CY} x2={CX + nx * (mid - gap)} y2={CY + ny * (mid - gap)}
                  stroke={line.col} strokeWidth={1.5} opacity={0.45} filter={`url(#${filter})`} />
                <line x1={CX + nx * (mid + gap)} y1={CY + ny * (mid + gap)} x2={nodes[i].x} y2={nodes[i].y}
                  stroke={line.col} strokeWidth={1.5} opacity={0.45} filter={`url(#${filter})`} />
                <circle cx={CX + nx * mid} cy={CY + ny * mid} r={3} fill={C.red} opacity={0.8} filter="url(#sf-glow-red)" />
              </g>
            )
          }
          return (
            <g key={i}>
              <path d={line.d} fill="none" stroke={line.glow} strokeWidth={5} opacity={0.2} filter={`url(#${filter})`} />
              <path d={line.d} fill="none" stroke={line.col} strokeWidth={1.2} opacity={0.55} className="sf-edge" />
            </g>
          )
        })}

        {/* core */}
        <g transform={`translate(${CX},${CY})`} className="sf-node">
          <circle r={36} fill="none" stroke={coreColor} strokeWidth={0.6} opacity={0.2} filter="url(#sf-core)" />
          <circle r={24} fill={`${coreColor}12`} stroke={coreColor} strokeWidth={1.8} filter="url(#sf-core)" />
          <circle r={14} fill="none" stroke={coreColor} strokeWidth={0.5} opacity={0.4} />
          <text textAnchor="middle" dominantBaseline="central" fill={coreColor} fontSize={10} fontWeight={800} letterSpacing="0.1em">
            FACILITY
          </text>
        </g>

        {/* room nodes */}
        {nodes.map((node, i) => {
          const filter = node.status === 'critical' ? 'sf-glow-red' : node.status === 'warning' ? 'sf-glow-orange' : 'sf-glow-neon'
          const r = 20 + node.driftContribution * 4
          return (
            <g key={node.id} transform={`translate(${node.x.toFixed(1)},${node.y.toFixed(1)})`} className="sf-node">
              <circle r={r + 10} fill="none" stroke={healthColor(node.status)} strokeWidth={0.7} opacity={0.25} filter={`url(#${filter})`} />
              <circle r={r} fill={`${healthColor(node.status)}12`} stroke={healthColor(node.status)} strokeWidth={1.6} filter={`url(#${filter})`} />
              <circle r={r - 6} fill="none" stroke={healthColor(node.status)} strokeWidth={0.4} opacity={0.35} />
              <text textAnchor="middle" dominantBaseline="central" fill={healthColor(node.status)} fontSize={9} fontWeight={800} letterSpacing="0.06em">
                {node.shortName}
              </text>
              {/* drift contribution indicator */}
              {node.driftContribution > 0.3 && (
                <circle r={r + 16} fill="none" stroke={healthColor(node.status)} strokeWidth={0.4} opacity={node.driftContribution * 0.2} filter={`url(#${filter})`} />
              )}
            </g>
          )
        })}
      </svg>

      {/* overlay label */}
      <div style={{
        position: 'absolute', top: 16, left: 18,
        fontSize: 8, letterSpacing: '0.16em', textTransform: 'uppercase',
        color: C.muted, fontWeight: 700,
      }}>
        System Coherence · {(coherence * 100).toFixed(0)}%
      </div>
    </div>
  )
}
