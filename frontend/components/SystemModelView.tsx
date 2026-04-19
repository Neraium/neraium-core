'use client'

import React from 'react'
import { DecisionUIState, Severity, DegradationStage } from '@/lib/decisionToUI'

interface Props {
  state: DecisionUIState
}

type NodeHealth = 'stable' | 'warning' | 'critical'

interface NodeDef {
  id: string
  label: string
  sublabel: string
  x: number
  y: number
  r: number
  type: 'primary' | 'hub' | 'sensor'
}

interface EdgeDef {
  from: string
  to: string
  type: 'loop' | 'control' | 'sensor'
}

/* ── Cannabis Grow Facility Topology ─────────────────────────────────────── */

const NODES: NodeDef[] = [
  /* rooms */
  { id: 'veg_a',  label: 'VEG-A',  sublabel: 'Vegetative A', x: 130, y: 100, r: 26, type: 'primary' },
  { id: 'veg_b',  label: 'VEG-B',  sublabel: 'Vegetative B', x: 130, y: 300, r: 26, type: 'primary' },
  { id: 'flow_a', label: 'FLOW-A', sublabel: 'Flowering A',  x: 462, y: 80,  r: 26, type: 'primary' },
  { id: 'flow_b', label: 'FLOW-B', sublabel: 'Flowering B',  x: 462, y: 200, r: 26, type: 'primary' },
  { id: 'flow_c', label: 'FLOW-C', sublabel: 'Flowering C',  x: 462, y: 320, r: 26, type: 'primary' },
  /* hub */
  { id: 'ctrl',   label: 'CTRL',   sublabel: 'Facility Control', x: 296, y: 200, r: 30, type: 'hub' },
  /* sensors */
  { id: 'env',    label: 'ENV',    sublabel: 'Climate',      x: 240, y: 340, r: 16, type: 'sensor' },
  { id: 'h2o',    label: 'H₂O',    sublabel: 'Irrigation',   x: 352, y: 340, r: 16, type: 'sensor' },
]

const EDGES: EdgeDef[] = [
  /* control → rooms */
  { from: 'ctrl', to: 'veg_a',  type: 'control' },
  { from: 'ctrl', to: 'veg_b',  type: 'control' },
  { from: 'ctrl', to: 'flow_a', type: 'control' },
  { from: 'ctrl', to: 'flow_b', type: 'control' },
  { from: 'ctrl', to: 'flow_c', type: 'control' },
  /* rooms ↔ sensors */
  { from: 'veg_a',  to: 'env', type: 'sensor' },
  { from: 'veg_b',  to: 'env', type: 'sensor' },
  { from: 'flow_a', to: 'env', type: 'sensor' },
  { from: 'flow_b', to: 'h2o', type: 'sensor' },
  { from: 'flow_c', to: 'h2o', type: 'sensor' },
  /* room-to-room airflow */
  { from: 'veg_a',  to: 'veg_b',  type: 'loop' },
  { from: 'flow_a', to: 'flow_b', type: 'loop' },
  { from: 'flow_b', to: 'flow_c', type: 'loop' },
]

const NODE_MAP = Object.fromEntries(NODES.map(n => [n.id, n]))

/* ── Health mapping ──────────────────────────────────────────────────────── */

function getNodeStates(severity: Severity, stage: DegradationStage): Record<string, NodeHealth> {
  const s = severity
  const st = stage

  if (s === Severity.LOW) {
    return { veg_a:'stable', veg_b:'stable', flow_a:'stable', flow_b:'stable', flow_c:'stable', ctrl:'stable', env:'stable', h2o:'stable' }
  }

  if (s === Severity.MODERATE) {
    return { veg_a:'stable', veg_b:'stable', flow_a:'warning', flow_b:'stable', flow_c:'stable', ctrl:'stable', env:'warning', h2o:'stable' }
  }

  if (s === Severity.ELEVATED) {
    return { veg_a:'stable', veg_b:'stable', flow_a:'warning', flow_b:'warning', flow_c:'stable', ctrl:'warning', env:'warning', h2o:'warning' }
  }

  /* HIGH / critical */
  if (st === DegradationStage.FAILURE_APPROACH) {
    return { veg_a:'warning', veg_b:'warning', flow_a:'critical', flow_b:'critical', flow_c:'critical', ctrl:'critical', env:'critical', h2o:'critical' }
  }
  return { veg_a:'warning', veg_b:'stable', flow_a:'critical', flow_b:'warning', flow_c:'stable', ctrl:'warning', env:'critical', h2o:'warning' }
}

const COLORS: Record<NodeHealth, { fill: string; stroke: string; glow: string; text: string; glowRadius: number }> = {
  stable:   { fill: 'rgba(126,159,46,0.12)',  stroke: '#7E9F2E', glow: 'rgba(126,159,46,0.25)',  text: '#A3E635', glowRadius: 8  },
  warning:  { fill: 'rgba(216,163,93,0.14)',  stroke: '#D8A35D', glow: 'rgba(216,163,93,0.30)',  text: '#FCD34D', glowRadius: 12 },
  critical: { fill: 'rgba(201,76,76,0.18)',   stroke: '#C94C4C', glow: 'rgba(201,76,76,0.35)',   text: '#FCA5A5', glowRadius: 16 },
}

/* ── Component ───────────────────────────────────────────────────────────── */

export function SystemModelView({ state }: Props) {
  const { statusHeader } = state
  const nodeStates = getNodeStates(statusHeader.severity, statusHeader.degradationStage)
  const severity = statusHeader.severity
  const stage = statusHeader.degradationStage

  const panelGlow =
    severity === Severity.HIGH     ? 'inset 0 0 80px rgba(201,76,76,0.08)'   :
    severity === Severity.ELEVATED ? 'inset 0 0 60px rgba(216,163,93,0.06)'  :
    severity === Severity.MODERATE ? 'inset 0 0 40px rgba(216,163,93,0.04)'  : 'none'

  const pulseSpeed =
    stage === DegradationStage.FAILURE_APPROACH ? '1.2s' :
    stage === DegradationStage.ACCELERATED      ? '1.6s' :
    stage === DegradationStage.PERSISTENT       ? '2.2s' :
    stage === DegradationStage.EMERGING         ? '2.8s' :
    stage === DegradationStage.EARLY_SHIFT      ? '3.5s' : '4.5s'

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative', overflow: 'hidden', boxShadow: panelGlow }}>
      {/* subtle grid background */}
      <div style={{
        position: 'absolute', inset: 0,
        backgroundImage: `
          linear-gradient(rgba(255,255,255,0.015) 1px, transparent 1px),
          linear-gradient(90deg, rgba(255,255,255,0.015) 1px, transparent 1px)
        `,
        backgroundSize: '40px 40px',
        maskImage: 'radial-gradient(ellipse at center, black 40%, transparent 80%)',
        WebkitMaskImage: 'radial-gradient(ellipse at center, black 40%, transparent 80%)',
      }} />

      <div style={{
        position: 'absolute', top: 14, left: 18, zIndex: 10,
        fontSize: 10, letterSpacing: '0.12em', textTransform: 'uppercase',
        color: '#4b5563', fontWeight: 700,
      }}>
        Facility Topology
      </div>

      <svg
        viewBox="0 0 592 400"
        width="100%" height="100%"
        preserveAspectRatio="xMidYMid meet"
        style={{ display: 'block' }}
      >
        <style>{`
          @keyframes nodePulse {
            0%, 100% { transform: scale(1); opacity: 1; }
            50%       { transform: scale(1.06); opacity: 0.92; }
          }
          @keyframes edgeFlow {
            0%   { stroke-dashoffset: 24; }
            100% { stroke-dashoffset: 0; }
          }
          @keyframes dataPacket {
            0%   { offset-distance: 0%; opacity: 0; }
            15%  { opacity: 1; }
            85%  { opacity: 1; }
            100% { offset-distance: 100%; opacity: 0; }
          }
          .node-group { transform-origin: center; animation: nodePulse ${pulseSpeed} ease-in-out infinite; }
          .edge-loop  { stroke-dasharray: 6 4; animation: edgeFlow 1.4s linear infinite; }
          .edge-ctrl   { stroke-dasharray: 3 3; animation: edgeFlow 2s linear infinite; }
          .edge-sensor { stroke-dasharray: 2 2; animation: edgeFlow 1s linear infinite; }
        `}</style>

        <defs>
          {(['stable','warning','critical'] as NodeHealth[]).map(h => (
            <filter key={h} id={`glow-${h}`} x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation={COLORS[h].glowRadius / 2.5} result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          ))}
        </defs>

        {/* Edges */}
        {EDGES.map((e, i) => {
          const a = NODE_MAP[e.from]
          const b = NODE_MAP[e.to]
          if (!a || !b) return null
          const healthA = nodeStates[e.from] ?? 'stable'
          const healthB = nodeStates[e.to] ?? 'stable'
          const health = healthA === 'critical' || healthB === 'critical' ? 'critical'
                       : healthA === 'warning'  || healthB === 'warning'  ? 'warning'
                       : 'stable'
          const c = COLORS[health]
          const dashClass = e.type === 'loop' ? 'edge-loop' : e.type === 'control' ? 'edge-ctrl' : 'edge-sensor'
          return (
            <g key={i}>
              <line
                x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                stroke={c.glow} strokeWidth={e.type === 'loop' ? 2.5 : e.type === 'control' ? 1.8 : 1.2}
                opacity={0.35}
                className={dashClass}
              />
              <line
                x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                stroke={c.stroke} strokeWidth={0.6}
                opacity={0.55}
              />
            </g>
          )
        })}

        {/* Nodes */}
        {NODES.map(n => {
          const health = nodeStates[n.id] ?? 'stable'
          const c = COLORS[health]
          const isHub = n.type === 'hub'
          return (
            <g key={n.id} transform={`translate(${n.x},${n.y})`} className="node-group">
              {/* outer glow ring */}
              <circle r={n.r + 10} fill="none" stroke={c.glow} strokeWidth={1.5} opacity={0.4} filter={`url(#glow-${health})`} />
              {/* core */}
              <circle r={n.r} fill={c.fill} stroke={c.stroke} strokeWidth={isHub ? 2.2 : 1.6} filter={`url(#glow-${health})`} />
              {/* hub inner ring */}
              {isHub && <circle r={n.r - 8} fill="none" stroke={c.stroke} strokeWidth={0.8} opacity={0.5} />}
              {/* label */}
              <text
                textAnchor="middle" dominantBaseline="central"
                fill={c.text} fontSize={isHub ? 11 : n.type === 'sensor' ? 8 : 10}
                fontWeight={800} letterSpacing="0.06em"
                fontFamily="-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif"
                style={{ pointerEvents: 'none', textShadow: `0 0 10px ${c.glow}` }}
              >
                {n.label}
              </text>
              {/* sublabel */}
              {n.sublabel && (
                <text
                  y={n.r + 14}
                  textAnchor="middle"
                  fill={c.text} fontSize={8} fontWeight={600} opacity={0.6}
                  fontFamily="-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif"
                  style={{ pointerEvents: 'none' }}
                >
                  {n.sublabel}
                </text>
              )}
            </g>
          )
        })}
      </svg>
    </div>
  )
}
