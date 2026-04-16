interface FrameData {
  structural_drift_score: number
  relational_stability_score: number
  event_admitted?: boolean
  [key: string]: any
}

interface ReplayChartProps {
  frames: FrameData[]
  currentIndex: number
}

export default function ReplayChart({ frames, currentIndex }: ReplayChartProps) {
  if (frames.length === 0) return null

  const W = 1200
  const H = 320
  const PX = 60
  const PY = 40
  const IW = W - 2 * PX
  const IH = H - 2 * PY

  const getDriftValue = (frame: FrameData) => Math.max(0, Math.min(1, frame.structural_drift_score || 0))
  const getStabilityValue = (frame: FrameData) => Math.max(0, Math.min(1, frame.relational_stability_score || 0))

  // Only show data up to current frame (live effect)
  const visibleFrames = frames.slice(0, currentIndex + 1)

  // Create data points for drift (only up to current index)
  const driftPoints = visibleFrames.map((frame, idx) => ({
    x: (idx / Math.max(frames.length - 1, 1)) * IW + PX,
    y: PY + IH - getDriftValue(frame) * IH,
    value: getDriftValue(frame),
  }))

  // Create data points for stability (only up to current index)
  const stabilityPoints = visibleFrames.map((frame, idx) => ({
    x: (idx / Math.max(frames.length - 1, 1)) * IW + PX,
    y: PY + IH - getStabilityValue(frame) * IH,
    value: getStabilityValue(frame),
  }))

  // Current frame position
  const currentX = (currentIndex / Math.max(frames.length - 1, 1)) * IW + PX

  // Build polyline points (only visible frames)
  const driftPath = driftPoints.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')
  const stabilityPath = stabilityPoints.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')

  // Build filled area paths for visual effect
  const driftAreaPath = driftPath ? `M ${driftPath} L ${PX + IW},${PY + IH} L ${PX},${PY + IH} Z` : ''
  const stabilityAreaPath = stabilityPath ? `M ${stabilityPath} L ${PX + IW},${PY + IH} L ${PX},${PY + IH} Z` : ''

  // Find flag events (changes in decision or admission status)
  const flagEvents: Array<{ index: number; x: number; admitted: boolean }> = []
  for (let i = 1; i < frames.length; i++) {
    const prev = frames[i - 1]
    const curr = frames[i]
    const prevAdmitted = prev.event_admitted || false
    const currAdmitted = curr.event_admitted || false

    // Mark when a decision changes
    if (prevAdmitted !== currAdmitted && i <= currentIndex) {
      const xPos = (i / Math.max(frames.length - 1, 1)) * IW + PX
      flagEvents.push({ index: i, x: xPos, admitted: currAdmitted })
    }
  }

  return (
    <div className="panel">
      <div className="panel-head">
        <span className="eyebrow">Replay Timeline</span>
        <span className="panel-subtitle">Structural drift and relational stability across {frames.length} frames</span>
      </div>
      <div className="chart-container">
        <svg className="chart-svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet">
          {/* Background */}
          <defs>
            <linearGradient id="driftAreaGrad" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="rgba(59, 130, 246, 0.2)" />
              <stop offset="100%" stopColor="rgba(59, 130, 246, 0.02)" />
            </linearGradient>
            <linearGradient id="stabilityAreaGrad" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="rgba(34, 197, 94, 0.2)" />
              <stop offset="100%" stopColor="rgba(34, 197, 94, 0.02)" />
            </linearGradient>
            <filter id="glow">
              <feGaussianBlur stdDeviation="2" result="coloredBlur" />
              <feMerge>
                <feMergeNode in="coloredBlur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Grid */}
          {[0.25, 0.5, 0.75].map((y) => (
            <line
              key={`grid-${y}`}
              x1={PX}
              y1={PY + IH - y * IH}
              x2={PX + IW}
              y2={PY + IH - y * IH}
              stroke="rgba(255,255,255,0.05)"
              strokeWidth="1"
            />
          ))}

          {/* Value markers */}
          {[0.25, 0.5, 0.75].map((y) => (
            <text
              key={`marker-${y}`}
              x={PX - 8}
              y={PY + IH - y * IH + 4}
              fill="rgba(255,255,255,0.3)"
              fontSize="10"
              textAnchor="end"
            >
              {(y * 100).toFixed(0)}%
            </text>
          ))}

          {/* Axis */}
          <line x1={PX} y1={PY} x2={PX} y2={PY + IH} stroke="rgba(255,255,255,0.2)" strokeWidth="1.5" />
          <line x1={PX} y1={PY + IH} x2={PX + IW} y2={PY + IH} stroke="rgba(255,255,255,0.2)" strokeWidth="1.5" />

          {/* Filled areas under lines */}
          {stabilityAreaPath && <path d={stabilityAreaPath} fill="url(#stabilityAreaGrad)" />}
          {driftAreaPath && <path d={driftAreaPath} fill="url(#driftAreaGrad)" />}

          {/* Stability line */}
          {stabilityPath && (
            <polyline
              points={stabilityPath}
              fill="none"
              stroke="#22C55E"
              strokeWidth="2.5"
              opacity="0.9"
              filter="url(#glow)"
            />
          )}

          {/* Drift line */}
          {driftPath && (
            <polyline
              points={driftPath}
              fill="none"
              stroke="#3B82F6"
              strokeWidth="2.5"
              opacity="1"
              filter="url(#glow)"
            />
          )}

          {/* Flag event markers */}
          {flagEvents.map((event, idx) => (
            <g key={`flag-${idx}`}>
              {/* Vertical line marking the flag */}
              <line
                x1={event.x}
                y1={PY}
                x2={event.x}
                y2={PY + IH}
                stroke={event.admitted ? '#22C55E' : '#F97316'}
                strokeWidth="2"
                opacity="0.5"
                strokeDasharray="3,2"
              />
              {/* Flag marker icon at top */}
              <g>
                {/* Flag background */}
                <circle
                  cx={event.x}
                  cy={PY - 16}
                  r="8"
                  fill={event.admitted ? '#22C55E' : '#F97316'}
                  opacity="0.9"
                />
                {/* Flag symbol */}
                <text
                  x={event.x}
                  y={PY - 12}
                  fill="#FFFFFF"
                  fontSize="12"
                  fontWeight="700"
                  textAnchor="middle"
                  dominantBaseline="middle"
                >
                  {event.admitted ? '✓' : '✕'}
                </text>
              </g>
              {/* Label */}
              <text
                x={event.x}
                y={PY - 32}
                fill={event.admitted ? '#22C55E' : '#F97316'}
                fontSize="9"
                fontWeight="600"
                textAnchor="middle"
                opacity="0.8"
                textTransform="uppercase"
              >
                {event.admitted ? 'ADMITTED' : 'SUPPRESSED'}
              </text>
            </g>
          ))}

          {/* Current position indicator */}
          <line
            x1={currentX}
            y1={PY}
            x2={currentX}
            y2={PY + IH}
            stroke="#06B6D4"
            strokeWidth="2"
            strokeDasharray="5,4"
            opacity="0.6"
          />
          {driftPoints[currentIndex] && (
            <circle
              cx={currentX}
              cy={driftPoints[currentIndex].y}
              r="7"
              fill="#3B82F6"
              stroke="#FFFFFF"
              strokeWidth="2"
              opacity="0.95"
            />
          )}
          {stabilityPoints[currentIndex] && (
            <circle
              cx={currentX}
              cy={stabilityPoints[currentIndex].y}
              r="7"
              fill="#22C55E"
              stroke="#FFFFFF"
              strokeWidth="2"
              opacity="0.95"
            />
          )}

          {/* Labels */}
          <text
            x={PX - 16}
            y={PY - 12}
            fill="rgba(255,255,255,0.7)"
            fontSize="12"
            fontWeight="600"
            textAnchor="end"
          >
            Signal
          </text>
          <text
            x={PX + IW + 8}
            y={PY + IH + 24}
            fill="rgba(255,255,255,0.7)"
            fontSize="12"
            fontWeight="600"
          >
            Timeline →
          </text>

          {/* Legend */}
          <g>
            <rect x={PX + 16} y={PY + 8} width={160} height={48} fill="rgba(0,0,0,0.4)" rx="4" />
            <line x1={PX + 24} y1={PY + 18} x2={PX + 44} y2={PY + 18} stroke="#22C55E" strokeWidth="2.5" />
            <text x={PX + 52} y={PY + 22} fill="rgba(255,255,255,0.85)" fontSize="11" fontWeight="600">
              Stability
            </text>

            <line x1={PX + 24} y1={PY + 38} x2={PX + 44} y2={PY + 38} stroke="#3B82F6" strokeWidth="2.5" />
            <text x={PX + 52} y={PY + 42} fill="rgba(255,255,255,0.85)" fontSize="11" fontWeight="600">
              Drift
            </text>
          </g>
        </svg>
      </div>
    </div>
  )
}
