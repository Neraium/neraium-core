interface TetrahedronPanelProps {
  frame: Record<string, unknown>
}

export default function TetrahedronPanel({ frame }: TetrahedronPanelProps) {
  const drift = ((frame.structural_drift_score as number) || 0)
  const stability = ((frame.relational_stability_score as number) || 0)
  const coherence = ((frame.coherence_score as number) || 0)
  const confidence = ((frame.confidence as number) || 0)

  // Tetrahedron visualization - simplified 2D projection
  const W = 300
  const H = 250
  const center = { x: W / 2, y: H / 2 }

  // Project 3D tetrahedron coordinates to 2D
  const scale = 60
  const vertices = [
    // Apex
    { x: center.x, y: center.y - 80, label: 'Coherence', value: coherence },
    // Base triangle
    { x: center.x - 70, y: center.y + 60, label: 'Drift', value: drift },
    { x: center.x + 70, y: center.y + 60, label: 'Stability', value: stability },
    { x: center.x, y: center.y + 80, label: 'Confidence', value: confidence },
  ]

  // Edges of tetrahedron
  const edges = [
    [0, 1],
    [0, 2],
    [0, 3],
    [1, 2],
    [2, 3],
    [1, 3],
  ]

  return (
    <div className="panel">
      <div className="panel-head">
        <span className="eyebrow">Structural State (Tetrahedron)</span>
        <span className="panel-subtitle">4-dimensional system state projection</span>
      </div>
      <div className="tetra-container">
        <div className="tetra-canvas">
          <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', maxWidth: '400px' }}>
            <defs>
              <radialGradient id="tetraGrad" cx="50%" cy="50%" r="70%">
                <stop offset="0%" stopColor="rgba(59, 130, 246, 0.1)" />
                <stop offset="100%" stopColor="rgba(5, 7, 15, 0.8)" />
              </radialGradient>
            </defs>

            {/* Background */}
            <rect width={W} height={H} fill="url(#tetraGrad)" />

            {/* Edges */}
            {edges.map((edge, idx) => {
              const v1 = vertices[edge[0]]
              const v2 = vertices[edge[1]]
              return (
                <line
                  key={`edge-${idx}`}
                  x1={v1.x}
                  y1={v1.y}
                  x2={v2.x}
                  y2={v2.y}
                  stroke="rgba(147, 197, 253, 0.3)"
                  strokeWidth="1.5"
                />
              )
            })}

            {/* Vertices */}
            {vertices.map((vertex, idx) => (
              <g key={`vertex-${idx}`}>
                {/* Glow */}
                <circle cx={vertex.x} cy={vertex.y} r={12} fill="rgba(59, 130, 246, 0.15)" />
                {/* Node */}
                <circle
                  cx={vertex.x}
                  cy={vertex.y}
                  r={8}
                  fill="#3B82F6"
                  stroke="#FFFFFF"
                  strokeWidth="1.5"
                  opacity={0.5 + vertex.value * 0.5}
                />
                {/* Label */}
                <text
                  x={vertex.x}
                  y={vertex.y + 20}
                  fill="rgba(203, 213, 225, 0.9)"
                  fontSize="10"
                  fontWeight="600"
                  textAnchor="middle"
                >
                  {vertex.label}
                </text>
              </g>
            ))}

            {/* Center point */}
            <circle cx={center.x} cy={center.y} r="3" fill="#06B6D4" />
          </svg>
        </div>

        <div className="tetra-info">
          <div className="tetra-label">
            <strong>Stability</strong>
            <div style={{
              marginTop: '4px',
              fontSize: '20px',
              fontWeight: 'bold',
              color: stability > 0.7 ? '#22C55E' : stability > 0.4 ? '#F97316' : '#EF4444',
              transition: 'all 0.3s ease-in-out',
              padding: '8px',
              backgroundColor: 'rgba(255,255,255,0.05)',
              borderRadius: '4px'
            }}>
              {(stability * 100).toFixed(0)}%
            </div>
          </div>
          <div className="tetra-label">
            <strong>Drift</strong>
            <div style={{
              marginTop: '4px',
              fontSize: '20px',
              fontWeight: 'bold',
              color: drift < 0.3 ? '#22C55E' : drift < 0.6 ? '#F97316' : '#EF4444',
              transition: 'all 0.3s ease-in-out',
              padding: '8px',
              backgroundColor: 'rgba(255,255,255,0.05)',
              borderRadius: '4px'
            }}>
              {(drift * 100).toFixed(0)}%
            </div>
          </div>
          <div className="tetra-label">
            <strong>Coherence</strong>
            <div style={{ marginTop: '4px', fontSize: '14px', color: '#06B6D4', transition: 'all 0.5s ease-in-out' }}>
              {(coherence * 100).toFixed(0)}%
            </div>
          </div>
          <div className="tetra-label">
            <strong>Confidence</strong>
            <div style={{ marginTop: '4px', fontSize: '14px', color: '#F97316', transition: 'all 0.5s ease-in-out' }}>
              {(confidence * 100).toFixed(0)}%
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
