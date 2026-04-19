'use client'

import React, { useEffect, useRef, useState, useCallback } from 'react'
import { motion } from 'framer-motion'
import { LiveSystem, LiveRoom } from '@/lib/simulation'
import { computeFacilityContext } from '@/lib/facilityContext'

interface Props {
  system: LiveSystem
  onSelectRoom: (id: string) => void
  hoveredRoomId?: string | null
  onHoverRoom?: (id: string | null) => void
}

/* ─── Palette ─────────────────────────────────────────────────────────────── */

const C = {
  bg: '#030405',
  neon: '#7cb342',
  neonGlow: 'rgba(124,179,66,0.35)',
  blue: '#5c9dff',
  blueGlow: 'rgba(92,157,255,0.30)',
  orange: '#ff9e6d',
  orangeGlow: 'rgba(255,158,109,0.35)',
  red: '#e05c5c',
  redGlow: 'rgba(224,92,92,0.40)',
  text: '#e8e9eb',
  secondary: '#8a8f9a',
  muted: '#3a3f4a',
  surface: '#080a0d',
}

/* ─── Helpers ─────────────────────────────────────────────────────────────── */

function healthColor(health: LiveRoom['status']) {
  return health === 'critical' ? C.red : health === 'warning' ? C.orange : C.neon
}

function healthGlow(health: LiveRoom['status']) {
  return health === 'critical' ? C.redGlow : health === 'warning' ? C.orangeGlow : C.neonGlow
}

function severityIdx(stage: string): number {
  return ['baseline', 'early_shift', 'emerging', 'persistent', 'accelerated', 'failure_approach'].indexOf(stage)
}

/* ─── Component ───────────────────────────────────────────────────────────── */

export function SystemCoherenceCore({ system, onSelectRoom, hoveredRoomId, onHoverRoom }: Props) {
  const { rooms, coherence, structuralDrift, stage } = system
  const idx = severityIdx(stage)
  const ctx = computeFacilityContext(system)

  /* animation tick */
  const [tick, setTick] = useState(0)
  const lastRef = useRef(0)
  useEffect(() => {
    let raf: number
    const loop = (t: number) => {
      if (t - lastRef.current > 50) {
        lastRef.current = t
        setTick(x => x + 1)
      }
      raf = requestAnimationFrame(loop)
    }
    raf = requestAnimationFrame(loop)
    return () => cancelAnimationFrame(raf)
  }, [])

  const time = tick * 0.035

  /* ── Geometry ──────────────────────────────────────────────────────────── */

  const VB = 600
  const CX = VB / 2
  const CY = VB / 2

  /* Node positions (5 rooms in pentagon) */
  const nodes = rooms.map((room, i) => {
    const angle = (i * 72 - 90) * (Math.PI / 180)
    const baseDist = 175
    const breathe = Math.sin(time + i * 1.2) * (idx === 0 ? 3 : idx <= 2 ? 7 : 14)
    const dist = baseDist + breathe

    /* drift deformation: pull high-drift nodes outward */
    const pull = room.driftContribution * 55
    const px = CX + Math.cos(angle) * (dist + pull)
    const py = CY + Math.sin(angle) * (dist + pull)

    /* critical jitter */
    const jx = idx >= 4 ? (Math.sin(time * 3.5 + i * 7.3) * 5) : 0
    const jy = idx >= 4 ? (Math.cos(time * 2.9 + i * 5.1) * 5) : 0

    return {
      ...room,
      x: px + jx,
      y: py + jy,
      angle,
    }
  })

  /* Inter-room tension lines (subsystem mesh) */
  const meshLines: Array<{
    d: string
    col: string
    opacity: number
    width: number
  }> = []
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const a = nodes[i]
      const b = nodes[j]
      const stress = (a.driftContribution + b.driftContribution) / 2
      const mx = (a.x + b.x) / 2
      const my = (a.y + b.y) / 2
      const perp = Math.atan2(b.y - a.y, b.x - a.x) + Math.PI / 2
      const bend = stress * 30 * (1 + idx * 0.3)
      const cpx = mx + Math.cos(perp) * bend + Math.sin(time * 0.7 + i + j) * 4
      const cpy = my + Math.sin(perp) * bend + Math.cos(time * 0.7 + i + j) * 4
      meshLines.push({
        d: `M${a.x.toFixed(1)},${a.y.toFixed(1)} Q${cpx.toFixed(1)},${cpy.toFixed(1)} ${b.x.toFixed(1)},${b.y.toFixed(1)}`,
        col: stress > 0.5 ? C.orange : stress > 0.25 ? C.blue : C.neon,
        opacity: 0.06 + stress * 0.18,
        width: 0.5 + stress * 1.2,
      })
    }
  }

  /* Core-to-node tension lines */
  const lines = nodes.map((node, i) => {
    const mx = (CX + node.x) / 2
    const my = (CY + node.y) / 2
    const perp = node.angle + Math.PI / 2
    const bend = (idx * 10 + structuralDrift * 30) * (node.driftContribution > 0.4 ? 1.8 : 0.6)
    const cpx = mx + Math.cos(perp) * bend + Math.sin(time + i) * 4
    const cpy = my + Math.sin(perp) * bend + Math.cos(time + i) * 4
    return {
      d: `M${CX.toFixed(1)},${CY.toFixed(1)} Q${cpx.toFixed(1)},${cpy.toFixed(1)} ${node.x.toFixed(1)},${node.y.toFixed(1)}`,
      col: healthColor(node.status),
      glow: healthGlow(node.status),
      broken: idx >= 4 && node.status === 'critical',
      stress: node.driftContribution,
    }
  })

  /* Coherence ring segments (reactive to system health) */
  const segments = 60
  const ringRadius = 115
  const ringPath = Array.from({ length: segments }, (_, i) => {
    const a = (i / segments) * Math.PI * 2
    const aNext = ((i + 1) / segments) * Math.PI * 2
    const x1 = CX + Math.cos(a) * ringRadius
    const y1 = CY + Math.sin(a) * ringRadius
    const x2 = CX + Math.cos(aNext) * ringRadius
    const y2 = CY + Math.sin(aNext) * ringRadius
    return { x1, y1, x2, y2, i }
  })

  /* Drift contours */
  const contours = idx >= 1 ? [1, 2, 3].map(layer => ({
    rx: 200 + layer * 35 + Math.sin(time * 0.5 + layer) * 12,
    ry: 155 + layer * 28 + Math.cos(time * 0.4 + layer) * 10,
    rot: layer * 15 + Math.sin(time * 0.3) * 6,
    opacity: (structuralDrift * 0.18) / layer,
    col: idx >= 3 ? C.orange : C.blue,
  })) : []

  /* Future path trace */
  const futureTrace = idx >= 1 ? (() => {
    const pts: Array<{ x: number; y: number }> = []
    for (let i = 0; i < 16; i++) {
      const t = i / 15
      const a = t * Math.PI * 1.6 - Math.PI / 2
      const r = 70 + t * 200 + structuralDrift * 60
      pts.push({
        x: CX + Math.cos(a + structuralDrift * 0.6) * r,
        y: CY + Math.sin(a + structuralDrift * 0.6) * r,
      })
    }
    return pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')
  })() : ''

  /* Core color based on severity */
  const coreColor =
    system.severity === 'HIGH' ? C.red
    : system.severity === 'ELEVATED' ? C.orange
    : system.severity === 'MODERATE' ? C.blue
    : C.neon

  /* Breathing speed based on severity */
  const breatheSpeed = idx >= 4 ? '0.8s' : idx >= 2 ? '1.5s' : '3.2s'
  const flowSpeed = idx >= 4 ? '0.5s' : '1.4s'

  /* Hover handler */
  const handleNodeHover = useCallback((id: string | null) => {
    onHoverRoom?.(id)
  }, [onHoverRoom])

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative', overflow: 'hidden' }}>
      {/* Ambient field gradient — shifts with system state */}
      <div style={{
        position: 'absolute', inset: 0,
        background: `radial-gradient(ellipse at 50% 50%, ${coreColor}0A 0%, transparent 55%)`,
        transition: 'background 1.5s ease',
      }} />

      {/* Deep field vignette */}
      <div style={{
        position: 'absolute', inset: 0,
        background: `radial-gradient(ellipse at 50% 50%, transparent 40%, ${C.bg} 100%)`,
        pointerEvents: 'none',
      }} />

      <svg viewBox={`0 0 ${VB} ${VB}`} width="100%" height="100%" preserveAspectRatio="xMidYMid meet" style={{ display: 'block' }}>
        <defs>
          <filter id="scc-glow-neon" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="6" /><feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge></filter>
          <filter id="scc-glow-blue" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="6" /><feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge></filter>
          <filter id="scc-glow-orange" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="6" /><feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge></filter>
          <filter id="scc-glow-red" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="8" /><feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge></filter>
          <filter id="scc-core" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="14" /><feMerge><feMergeNode /><feMergeNode in="SourceGraphic" /></feMerge></filter>
          <filter id="scc-field" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="20" /><feMerge><feMergeNode /></feMerge></filter>
        </defs>

        <style>{`
          @keyframes sccBreathe { 0%,100%{transform:scale(1)} 50%{transform:scale(1.06)} }
          @keyframes sccFlow { 0%{stroke-dashoffset:24} 100%{stroke-dashoffset:0} }
          @keyframes sccPulse { 0%,100%{opacity:0.35} 50%{opacity:0.7} }
          .scc-node { transform-origin:center; animation: sccBreathe ${breatheSpeed} ease-in-out infinite; }
          .scc-edge { stroke-dasharray: 4 4; animation: sccFlow ${flowSpeed} linear infinite; }
          .scc-core-ring { animation: sccPulse ${breatheSpeed} ease-in-out infinite; }
        `}</style>

        {/* Subtle grid field */}
        <g opacity={0.04}>
          {Array.from({ length: 14 }, (_, i) => (
            <line key={`v${i}`} x1={(i + 1) * (VB / 15)} y1={0} x2={(i + 1) * (VB / 15)} y2={VB} stroke={C.muted} strokeWidth={0.4} />
          ))}
          {Array.from({ length: 10 }, (_, i) => (
            <line key={`h${i}`} x1={0} y1={(i + 1) * (VB / 11)} x2={VB} y2={(i + 1) * (VB / 11)} stroke={C.muted} strokeWidth={0.4} />
          ))}
        </g>

        {/* Drift contours */}
        {contours.map((c, i) => (
          <ellipse key={i} cx={CX} cy={CY} rx={c.rx} ry={c.ry}
            fill="none" stroke={c.col} strokeWidth={0.7} opacity={c.opacity}
            transform={`rotate(${c.rot.toFixed(1)} ${CX} ${CY})`}
          />
        ))}

        {/* Future path trace */}
        {futureTrace && (
          <path d={futureTrace} fill="none" stroke={C.blue} strokeWidth={1.2} opacity={0.1} strokeDasharray="5 7" />
        )}

        {/* Coherence ring — reactive to health */}
        {ringPath.map(seg => {
          const segCoherence = coherence - Math.abs(Math.sin(seg.i / segments * Math.PI * 2 + time * 0.3)) * (1 - coherence)
          const opacity = Math.max(0, segCoherence * 0.6)
          const col = segCoherence < 0.3 ? C.red : segCoherence < 0.6 ? C.orange : C.neon
          return (
            <line key={seg.i} x1={seg.x1} y1={seg.y1} x2={seg.x2} y2={seg.y2}
              stroke={col} strokeWidth={1.4} opacity={opacity}
              className="scc-core-ring"
            />
          )
        })}

        {/* Inter-room mesh lines */}
        {meshLines.map((line, i) => (
          <path key={`mesh-${i}`} d={line.d} fill="none" stroke={line.col} strokeWidth={line.width} opacity={line.opacity} />
        ))}

        {/* Core-to-node tension lines */}
        {lines.map((line, i) => {
          const filter = line.col === C.red ? 'scc-glow-red' : line.col === C.orange ? 'scc-glow-orange' : line.col === C.blue ? 'scc-glow-blue' : 'scc-glow-neon'
          if (line.broken) {
            const dx = nodes[i].x - CX
            const dy = nodes[i].y - CY
            const len = Math.sqrt(dx * dx + dy * dy)
            const nx = dx / len
            const ny = dy / len
            const gap = 22
            const mid = len / 2
            return (
              <g key={i}>
                <line x1={CX} y1={CY} x2={CX + nx * (mid - gap)} y2={CY + ny * (mid - gap)}
                  stroke={line.col} strokeWidth={2} opacity={0.5} filter={`url(#${filter})`} />
                <line x1={CX + nx * (mid + gap)} y1={CY + ny * (mid + gap)} x2={nodes[i].x} y2={nodes[i].y}
                  stroke={line.col} strokeWidth={2} opacity={0.5} filter={`url(#${filter})`} />
                <circle cx={CX + nx * mid} cy={CY + ny * mid} r={4} fill={C.red} opacity={0.9} filter="url(#scc-glow-red)" />
                {/* Fracture sparks */}
                <circle cx={CX + nx * mid} cy={CY + ny * mid} r={8} fill="none" stroke={C.red} strokeWidth={0.5} opacity={0.3}>
                  <animate attributeName="r" values="8;14;8" dur="1.5s" repeatCount="indefinite" />
                  <animate attributeName="opacity" values="0.3;0.1;0.3" dur="1.5s" repeatCount="indefinite" />
                </circle>
              </g>
            )
          }
          return (
            <g key={i}>
              <path d={line.d} fill="none" stroke={line.glow} strokeWidth={7} opacity={0.18} filter={`url(#${filter})`} />
              <path d={line.d} fill="none" stroke={line.col} strokeWidth={1.4} opacity={0.6} className="scc-edge" />
              {/* Stress highlight on hover */}
              {hoveredRoomId === nodes[i].id && (
                <path d={line.d} fill="none" stroke={line.col} strokeWidth={3} opacity={0.35} filter={`url(#${filter})`} />
              )}
            </g>
          )
        })}

        {/* System Coherence Core */}
        <g transform={`translate(${CX},${CY})`} className="scc-node">
          {/* Outer resonance field */}
          <circle r={52} fill="none" stroke={coreColor} strokeWidth={0.4} opacity={0.12} filter="url(#scc-field)">
            <animate attributeName="r" values="52;58;52" dur={breatheSpeed} repeatCount="indefinite" />
          </circle>
          {/* Mid ring */}
          <circle r={42} fill="none" stroke={coreColor} strokeWidth={0.6} opacity={0.18} filter="url(#scc-core)" />
          {/* Main core body */}
          <circle r={28} fill={`${coreColor}14`} stroke={coreColor} strokeWidth={2.2} filter="url(#scc-core)" />
          {/* Inner ring */}
          <circle r={16} fill="none" stroke={coreColor} strokeWidth={0.6} opacity={0.45} />
          {/* Core pulse dot */}
          <circle r={5} fill={coreColor} opacity={0.8} filter="url(#scc-core)">
            <animate attributeName="opacity" values="0.6;1;0.6" dur={breatheSpeed} repeatCount="indefinite" />
          </circle>
        </g>

        {/* Room nodes */}
        {nodes.map((node, i) => {
          const filter = node.status === 'critical' ? 'scc-glow-red' : node.status === 'warning' ? 'scc-glow-orange' : 'scc-glow-neon'
          const r = 24 + node.driftContribution * 6
          const isHovered = hoveredRoomId === node.id
          return (
            <g
              key={node.id}
              transform={`translate(${node.x.toFixed(1)},${node.y.toFixed(1)})`}
              className="scc-node"
              style={{ cursor: 'pointer' }}
              onMouseEnter={() => handleNodeHover(node.id)}
              onMouseLeave={() => handleNodeHover(null)}
              onClick={() => onSelectRoom(node.id)}
            >
              {/* Hover halo */}
              {isHovered && (
                <circle r={r + 24} fill="none" stroke={healthColor(node.status)} strokeWidth={0.8} opacity={0.15} filter={`url(#${filter})`}>
                  <animate attributeName="r" values={`${r + 24};${r + 32};${r + 24}`} dur="1.5s" repeatCount="indefinite" />
                </circle>
              )}
              {/* Outer glow ring */}
              <circle r={r + 12} fill="none" stroke={healthColor(node.status)} strokeWidth={0.8} opacity={0.22} filter={`url(#${filter})`} />
              {/* Main node body */}
              <circle r={r} fill={`${healthColor(node.status)}10`} stroke={healthColor(node.status)} strokeWidth={1.8} filter={`url(#${filter})`} />
              {/* Inner detail ring */}
              <circle r={r - 8} fill="none" stroke={healthColor(node.status)} strokeWidth={0.45} opacity={0.4} />
              {/* Label */}
              <text textAnchor="middle" dominantBaseline="central" fill={healthColor(node.status)} fontSize={10} fontWeight={800} letterSpacing="0.08em">
                {node.shortName}
              </text>
              {/* Drift contribution indicator */}
              {node.driftContribution > 0.25 && (
                <circle r={r + 20} fill="none" stroke={healthColor(node.status)} strokeWidth={0.5} opacity={node.driftContribution * 0.25} filter={`url(#${filter})`}>
                  <animate attributeName="opacity" values={`${node.driftContribution * 0.15};${node.driftContribution * 0.3};${node.driftContribution * 0.15}`} dur="2s" repeatCount="indefinite" />
                </circle>
              )}
            </g>
          )
        })}
      </svg>

      {/* Overlay label */}
      <div style={{
        position: 'absolute', top: 18, left: 20,
        fontSize: 8, letterSpacing: '0.18em', textTransform: 'uppercase',
        color: C.muted, fontWeight: 700,
      }}>
        System Coherence Engine
      </div>

      <div style={{
        position: 'absolute', top: 18, right: 20,
        fontSize: 8, letterSpacing: '0.12em',
        color: C.secondary, fontWeight: 700, fontVariantNumeric: 'tabular-nums',
      }}>
        {(coherence * 100).toFixed(0)}% · {ctx.coherenceTrend}
      </div>

      {/* Bottom-left: stage label */}
      <div style={{
        position: 'absolute', bottom: 18, left: 20,
        display: 'flex', alignItems: 'center', gap: 10,
      }}>
        <span style={{
          width: 8, height: 8, borderRadius: '50%',
          background: coreColor,
          boxShadow: `0 0 12px ${coreColor}40`,
        }} />
        <span style={{ fontSize: 9, color: C.secondary, fontWeight: 700, letterSpacing: '0.08em', textTransform: 'uppercase' }}>
          {stage.replace('_', ' ')}
        </span>
      </div>

      {/* Bottom-right: structural drift */}
      <div style={{
        position: 'absolute', bottom: 18, right: 20,
        fontSize: 8, color: C.muted, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase',
      }}>
        Structural Drift {(structuralDrift * 100).toFixed(0)}%
      </div>
    </div>
  )
}
