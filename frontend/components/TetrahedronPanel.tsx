interface TetrahedronPanelProps {
  frame: Record<string, unknown>
}

export default function TetrahedronPanel({ frame }: TetrahedronPanelProps) {
  const drift = Math.max(0, Math.min(1, (frame.structural_drift_score as number) || 0))
  const stability = Math.max(0, Math.min(1, (frame.relational_stability_score as number) || 0))
  const coherence = Math.max(0, Math.min(1, (frame.coherence_score as number) || 0))
  const confidence = Math.max(0, Math.min(1, (frame.confidence as number) || 0))

  // Tetrahedron visualization - improved 2D projection
  const W = 360
  const H = 300
  const center = { x: W / 2, y: H / 2 + 10 }

  // Regular tetrahedron vertices in 2D projection
  // Using a better geometric layout
  const vertices = [
    // Top apex
    { x: center.x, y: center.y - 90, label: 'Coherence', value: coherence, color: '#06B6D4' },
    // Base left
    { x: center.x - 90, y: center.y + 70, label: 'Drift', value: drift, color: '#3B82F6' },
    // Base right
    { x: center.x + 90, y: center.y + 70, label: 'Stability', value: stability, color: '#22C55E' },
    // Base bottom
    { x: center.x, y: center.y + 95, label: 'Confidence', value: confidence, color: '#F97316' },
  ]

  // Edges of tetrahedron (all pairs)
  const edges = [
    [0, 1], // apex to left
    [0, 2], // apex to right
    [0, 3], // apex to bottom
    [1, 2], // left to right
    [1, 3], // left to bottom
    [2, 3], // right to bottom
  ]

  return (
    <div className="panel">
      <div className="panel-head">
        <span className="eyebrow">Structural State (Tetrahedron)</span>
        <span className="panel-subtitle">4-dimensional system state projection</span>
      </div>
      <div className="tetra-container">
        <div className="tetra-canvas">
          <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet" style={{ width: '100%' }}>
            <defs>
              <radialGradient id="tetraGrad" cx="50%" cy="45%" r="65%">
                <stop offset="0%" stopColor="rgba(59, 130, 246, 0.08)" />
                <stop offset="100%" stopColor="rgba(5, 7, 15, 0.6)" />
              </radialGradient>
              <filter id="tetraGlow">
                <feGaussianBlur stdDeviation="1.5" result="coloredBlur" />
                <feMerge>
                  <feMergeNode in="coloredBlur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>

            {/* Background */}
            <rect width={W} height={H} fill="url(#tetraGrad)" />

            {/* Edges with gradient effect */}
            {edges.map((edge, idx) => {
              const v1 = vertices[edge[0]]
              const v2 = vertices[edge[1]]
              const edgeColor = idx < 3 ? 'rgba(96, 165, 250, 0.4)' : 'rgba(96, 165, 250, 0.25)'
              return (
                <line
                  key={`edge-${idx}`}
                  x1={v1.x}
                  y1={v1.y}
                  x2={v2.x}
                  y2={v2.y}
                  stroke={edgeColor}
                  strokeWidth="1.5"
                  opacity="0.7"
                />
              )
            })}

            {/* Vertices */}
            {vertices.map((vertex, idx) => {
              const displayRadius = 5 + vertex.value * 5 // Scale glow with value
              return (
                <g key={`vertex-${idx}`}>
                  {/* Outer glow */}
                  <circle
                    cx={vertex.x}
                    cy={vertex.y}
                    r={displayRadius + 6}
                    fill={vertex.color}
                    opacity={0.1 + vertex.value * 0.15}
                  />
                  {/* Middle glow */}
                  <circle
                    cx={vertex.x}
                    cy={vertex.y}
                    r={displayRadius + 2}
                    fill={vertex.color}
                    opacity={0.15 + vertex.value * 0.2}
                  />
                  {/* Core node */}
                  <circle
                    cx={vertex.x}
                    cy={vertex.y}
                    r={displayRadius}
                    fill={vertex.color}
                    stroke="#FFFFFF"
                    strokeWidth="2"
                    opacity={0.6 + vertex.value * 0.4}
                    filter="url(#tetraGlow)"
                  />
                  {/* Label */}
                  <text
                    x={vertex.x}
                    y={vertex.y + 26}
                    fill="rgba(203, 213, 225, 0.85)"
                    fontSize="11"
                    fontWeight="600"
                    textAnchor="middle"
                  >
                    {vertex.label}
                  </text>
                  {/* Value indicator */}
                  <text
                    x={vertex.x}
                    y={vertex.y + 40}
                    fill={vertex.color}
                    fontSize="10"
                    fontWeight="700"
                    textAnchor="middle"
                    opacity="0.8"
                  >
                    {(vertex.value * 100).toFixed(0)}%
                  </text>
                </g>
              )
            })}

            {/* Center point */}
            <circle cx={center.x} cy={center.y} r="2.5" fill="rgba(255,255,255,0.4)" />
          </svg>
        </div>

        <div className="tetra-info">
          <div className="tetra-label" style={{ borderLeftColor: '#3B82F6' }}>
            <strong>Drift</strong>
            <div style={{ marginTop: '4px', fontSize: '16px', color: '#3B82F6', transition: 'all 0.5s ease-in-out' }}>
              {(drift * 100).toFixed(1)}%
            </div>
          </div>
          <div className="tetra-label" style={{ borderLeftColor: '#22C55E' }}>
            <strong>Stability</strong>
            <div style={{ marginTop: '4px', fontSize: '16px', color: '#22C55E', transition: 'all 0.5s ease-in-out' }}>
              {(stability * 100).toFixed(1)}%
            </div>
          </div>
          <div className="tetra-label" style={{ borderLeftColor: '#06B6D4' }}>
            <strong>Coherence</strong>
            <div style={{ marginTop: '4px', fontSize: '16px', color: '#06B6D4', transition: 'all 0.5s ease-in-out' }}>
              {(coherence * 100).toFixed(1)}%
            </div>
          </div>
          <div className="tetra-label" style={{ borderLeftColor: '#F97316' }}>
            <strong>Confidence</strong>
            <div style={{ marginTop: '4px', fontSize: '16px', color: '#F97316', transition: 'all 0.5s ease-in-out' }}>
              {(confidence * 100).toFixed(1)}%
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
