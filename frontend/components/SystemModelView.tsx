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

const NODES: NodeDef[] = [
  { id: 'compressor', label: 'COMP', sublabel: 'Compressor',   x: 108, y: 200, r: 26, type: 'primary' },
  { id: 'condenser',  label: 'COND', sublabel: 'Condenser',    x: 296, y: 68,  r: 26, type: 'primary' },
  { id: 'expansion',  label: 'EXP',  sublabel: 'Expansion',    x: 484, y: 200, r: 26, type: 'primary' },
  { id: 'evaporator', label: 'EVAP', sublabel: 'Evaporator',   x: 296, y: 332, r: 26, type: 'primary' },
  { id: 'control',    label: 'CTRL', sublabel: 'Control Unit', x: 296, y: 200, r: 30, type: 'hub' },
  { id: 'temp_a',     label: 'T-A',  sublabel: '',             x: 174, y: 114, r: 13, type: 'sensor' },
  { id: 'flow_b',     label: 'F-B',  sublabel: '',             x: 418, y: 114, r: 13, type: 'sensor' },
  { id: 'pressure',   label: 'P-01', sublabel: '',             x: 178, y: 296, r: 13, type: 'sensor' },
]

const EDGES: EdgeDef[] = [
  { from: 'compressor', to: 'condenser',  type: 'loop'    },
  { from: 'condenser',  to: 'expansion',  type: 'loop'    },
  { from: 'expansion',  to: 'evaporator', type: 'loop'    },
  { from: 'evaporator', to: 'compressor', type: 'loop'    },
  { from: 'control',    to: 'compressor', type: 'control' },
  { from: 'control',    to: 'condenser',  type: 'control' },
  { from: 'control',    to: 'expansion',  type: 'control' },
  { from: 'control',    to: 'evaporator', type: 'control' },
  { from: 'temp_a',     to: 'condenser',  type: 'sensor'  },
  { from: 'flow_b',     to: 'expansion',  type: 'sensor'  },
  { from: 'pressure',   to: 'compressor', type: 'sensor'  },
]

const NODE_MAP = Object.fromEntries(NODES.map(n => [n.id, n]))

function getNodeStates(severity: Severity): Record<string, NodeHealth> {
  if (severity === Severity.LOW) {
    return { compressor:'stable', condenser:'stable', expansion:'stable', evaporator:'stable', control:'stable', temp_a:'stable', flow_b:'stable', pressure:'stable' }
  }
  if (severity === Severity.MODERATE) {
    return { compressor:'warning', condenser:'stable', expansion:'stable', evaporator:'stable', control:'stable', temp_a:'stable', flow_b:'stable', pressure:'warning' }
  }
  if (severity === Severity.ELEVATED) {
    return { compressor:'warning', condenser:'stable', expansion:'warning', evaporator:'stable', control:'warning', temp_a:'warning', flow_b:'stable', pressure:'warning' }
  }
  return { compressor:'critical', condenser:'warning', expansion:'critical', evaporator:'warning', control:'critical', temp_a:'warning', flow_b:'warning', pressure:'critical' }
}

const COLORS: Record<NodeHealth, { fill: string; stroke: string; glow: string; text: string; glowRadius: number }> = {
  stable:   { fill: 'rgba(34,197,94,0.11)',   stroke: '#22c55e', glow: 'rgba(34,197,94,0.22)',   text: '#86efac', glowRadius: 8  },
  warning:  { fill: 'rgba(245,158,11,0.14)',  stroke: '#f59e0b', glow: 'rgba(245,158,11,0.28)',  text: '#fcd34d', glowRadius: 11 },
  critical: { fill: 'rgba(239,68,68,0.18)',   stroke: '#ef4444', glow: 'rgba(239,68,68,0.32)',   text: '#fca5a5', glowRadius: 15 },
}

export function SystemModelView({ state }: Props) {
  const { statusHeader } = state
  const nodeStates = getNodeStates(statusHeader.severity)
  const severity = statusHeader.severity

  const panelGlow =
    severity === Severity.HIGH     ? 'inset 0 0 80px rgba(239,68,68,0.07)'     :
    severity === Severity.ELEVATED ? 'inset 0 0 60px rgba(245,158,11,0.06)'    :
    severity === Severity.MODERATE ? 'inset 0 0 40px rgba(245,158,11,0.03)'    : 'none'

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative', overflow: 'hidden', boxShadow: panelGlow }}>
      <div style={{
        position: 'absolute', top: 14, left: 18, zIndex: 10,
        fontSize: 10, letterSpacing: '0.1em', textTransform: 'uppercase',
        color: '#4b5563', fontWeight: 700,
      }}>
        System Model
      </div>

      <svg
        viewBox="0 0 592 400"
        width="100%" height="100%"
        preserveAspectRatio="xMidYMid meet"
        style={{ display: 'block' }}
      >
        <style>{`
          @keyframes smStablePulse   { 0%,100%{opacity:.22} 50%{opacity:.45} }
          @keyframes smWarningPulse  { 0%,100%{opacity:.28} 50%{opacity:.65} }
          @keyframes smCriticalPulse { 0%,100%{opacity:.32} 50%{opacity:.82} }
          @keyframes smEdgeFlow      { 0%{stroke-dashoffset:24} 100%{stroke-dashoffset:0} }
          @keyframes smCritEdgeFlow  { 0%{stroke-dashoffset:16} 100%{stroke-dashoffset:0} }
          .sm-glow-stable   { animation: smStablePulse   4.2s ease-in-out infinite }
          .sm-glow-warning  { animation: smWarningPulse  2.2s ease-in-out infinite }
          .sm-glow-critical { animation: smCriticalPulse 1.1s ease-in-out infinite }
          .sm-edge-warn     { animation: smEdgeFlow      2.8s linear infinite }
          .sm-edge-crit     { animation: smCritEdgeFlow  1.2s linear infinite }
        `}</style>

        <defs>
          <pattern id="smGrid" width="38" height="38" patternUnits="userSpaceOnUse">
            <path d="M38,0 L0,0 L0,38" fill="none" stroke="rgba(255,255,255,0.022)" strokeWidth="0.5"/>
          </pattern>
        </defs>

        {/* background grid */}
        <rect width="592" height="400" fill="url(#smGrid)" />

        {/* Edges */}
        {EDGES.map(edge => {
          const from = NODE_MAP[edge.from]
          const to   = NODE_MAP[edge.to]
          const fh   = nodeStates[edge.from]
          const th   = nodeStates[edge.to]
          const eh: NodeHealth =
            (fh === 'critical' || th === 'critical') ? 'critical' :
            (fh === 'warning'  || th === 'warning')  ? 'warning'  : 'stable'

          const c  = COLORS[eh]
          const sw = edge.type === 'loop' ? 1.8 : edge.type === 'control' ? 1.2 : 0.9
          const op = eh === 'stable' ? 0.28 : eh === 'warning' ? 0.6 : 0.85
          const da = eh !== 'stable'
            ? (eh === 'critical' ? '7 4' : '9 5')
            : (edge.type !== 'loop' ? '5 4' : undefined)

          return (
            <g key={`${edge.from}-${edge.to}`}>
              {eh !== 'stable' && (
                <line x1={from.x} y1={from.y} x2={to.x} y2={to.y}
                  stroke={c.stroke} strokeWidth={sw + 7} strokeOpacity={0.06} />
              )}
              <line
                x1={from.x} y1={from.y} x2={to.x} y2={to.y}
                stroke={c.stroke} strokeWidth={sw} strokeOpacity={op}
                strokeDasharray={da}
                className={eh === 'critical' ? 'sm-edge-crit' : eh === 'warning' ? 'sm-edge-warn' : undefined}
              />
            </g>
          )
        })}

        {/* Nodes */}
        {NODES.map((node, idx) => {
          const h  = nodeStates[node.id]
          const c  = COLORS[h]
          const delay = `${idx * 0.38}s`

          return (
            <g key={node.id}>
              {/* glow halo */}
              <circle
                cx={node.x} cy={node.y}
                r={node.r + c.glowRadius}
                fill={c.glow}
                className={`sm-glow-${h}`}
                style={{ animationDelay: delay }}
              />
              {/* body */}
              <circle
                cx={node.x} cy={node.y} r={node.r}
                style={{
                  fill: c.fill,
                  stroke: c.stroke,
                  strokeWidth: node.type === 'hub' ? 2 : 1.5,
                  strokeOpacity: h === 'stable' ? 0.55 : 0.92,
                  transition: 'fill 0.9s ease, stroke 0.9s ease',
                }}
              />
              {/* primary label */}
              <text
                x={node.x} y={node.y}
                textAnchor="middle" dominantBaseline="central"
                fill={c.text} fontSize={node.type === 'sensor' ? 7 : 8.5}
                fontWeight="700" letterSpacing="0.06em"
                fontFamily="-apple-system,'Segoe UI',sans-serif"
                style={{ transition: 'fill 0.9s ease' }}
              >
                {node.label}
              </text>
              {/* sublabel */}
              {node.sublabel && (
                <text
                  x={node.x} y={node.y + node.r + 11}
                  textAnchor="middle"
                  fill="rgba(107,114,128,0.8)" fontSize="6.5"
                  letterSpacing="0.08em"
                  fontFamily="-apple-system,'Segoe UI',sans-serif"
                >
                  {node.sublabel}
                </text>
              )}
            </g>
          )
        })}
      </svg>
    </div>
  )
}
