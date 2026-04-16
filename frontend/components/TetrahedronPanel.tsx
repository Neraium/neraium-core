interface TetrahedronPanelProps {
  frame: Record<string, unknown>
}

export default function TetrahedronPanel({ frame }: TetrahedronPanelProps) {
  // The 4 dimensions of the tetrahedron (system state geometry)
  const structuralDrift = ((frame.structural_drift_score as number) || 0)
  const relationalStability = ((frame.relational_stability_score as number) || 0)
  const temporalConsistency = ((frame.temporal_consistency_score as number) || 0)
  const systemCoherence = ((frame.system_coherence as number) || 0)
  const compositeInstability = ((frame.composite_instability as number) || 0)
  const systemHealth = ((frame.system_health_metric as number) || 0)

  // Tetrahedron visualization - 4D state space projection
  const W = 300
  const H = 250
  const center = { x: W / 2, y: H / 2 }

  // Project 4D tetrahedron to 2D (using isometric-like projection)
  const vertices = [
    // Vertex 1: Structural Drift
    { x: center.x, y: center.y - 80, label: 'Drift', value: structuralDrift, color: '#3B82F6' },
    // Vertex 2: Relational Stability
    { x: center.x - 70, y: center.y + 60, label: 'Stability', value: relationalStability, color: '#22C55E' },
    // Vertex 3: Temporal Consistency
    { x: center.x + 70, y: center.y + 60, label: 'Temporal', value: temporalConsistency, color: '#06B6D4' },
    // Vertex 4: System Coherence
    { x: center.x, y: center.y + 80, label: 'Coherence', value: systemCoherence, color: '#F97316' },
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
                <circle cx={vertex.x} cy={vertex.y} r={12} fill={vertex.color} opacity="0.15" />
                {/* Node */}
                <circle
                  cx={vertex.x}
                  cy={vertex.y}
                  r={8}
                  fill={vertex.color}
                  stroke="#FFFFFF"
                  strokeWidth="1.5"
                  opacity={0.5 + vertex.value * 0.5}
                />
                {/* Label */}
                <text
                  x={vertex.x}
                  y={vertex.y + 20}
                  fill={vertex.color}
                  fontSize="11"
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
          <div style={{ marginBottom: '12px', paddingBottom: '12px', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
            <div style={{ fontSize: '12px', color: 'rgba(255,255,255,0.6)', marginBottom: '4px' }}>TETRAHEDRON STATE</div>
            <div style={{ fontSize: '13px', fontWeight: '600', color: compositeInstability > 0.7 ? '#EF4444' : compositeInstability > 0.5 ? '#F97316' : '#22C55E' }}>
              Risk: {compositeInstability > 0.7 ? 'CRITICAL' : compositeInstability > 0.5 ? 'HIGH' : compositeInstability > 0.3 ? 'MEDIUM' : 'LOW'}
            </div>
          </div>

          <div className="tetra-label">
            <strong style={{ color: '#3B82F6' }}>Structural Drift</strong>
            <div style={{ marginTop: '4px', fontSize: '18px', fontWeight: 'bold', color: '#3B82F6', transition: 'all 0.3s ease-in-out' }}>
              {(structuralDrift * 100).toFixed(0)}%
            </div>
          </div>

          <div className="tetra-label">
            <strong style={{ color: '#22C55E' }}>Relational Stability</strong>
            <div style={{ marginTop: '4px', fontSize: '18px', fontWeight: 'bold', color: '#22C55E', transition: 'all 0.3s ease-in-out' }}>
              {(relationalStability * 100).toFixed(0)}%
            </div>
          </div>

          <div className="tetra-label">
            <strong style={{ color: '#06B6D4' }}>Temporal Consistency</strong>
            <div style={{ marginTop: '4px', fontSize: '14px', color: '#06B6D4', transition: 'all 0.3s ease-in-out' }}>
              {(temporalConsistency * 100).toFixed(0)}%
            </div>
          </div>

          <div className="tetra-label">
            <strong style={{ color: '#F97316' }}>System Coherence</strong>
            <div style={{ marginTop: '4px', fontSize: '14px', color: '#F97316', transition: 'all 0.3s ease-in-out' }}>
              {(systemCoherence * 100).toFixed(0)}%
            </div>
          </div>

          <div className="tetra-label" style={{ marginTop: '8px', paddingTop: '8px', borderTop: '1px solid rgba(255,255,255,0.1)' }}>
            <strong>System Health</strong>
            <div style={{
              marginTop: '4px',
              fontSize: '16px',
              fontWeight: 'bold',
              color: systemHealth > 0.7 ? '#22C55E' : systemHealth > 0.4 ? '#F97316' : '#EF4444',
              transition: 'all 0.3s ease-in-out'
            }}>
              {(systemHealth * 100).toFixed(0)}%
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
