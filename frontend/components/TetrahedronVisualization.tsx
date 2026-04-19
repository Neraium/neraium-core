'use client'

import { useEffect, useRef } from 'react'

interface TetrahedronVisualizationProps {
  state: any
}

export function TetrahedronVisualization({ state }: TetrahedronVisualizationProps) {
  const svgRef = useRef<SVGSVGElement>(null)

  useEffect(() => {
    if (!svgRef.current || !state) return

    // Calculate deformation based on drift
    const drift = state.drift || 0
    const coherence = state.coherence || 0.75
    const totalDeformation = Math.min(drift * 0.3, 0.3)

    // Core radius shrinks under stress
    const coreRadius = Math.max(0.08, 0.15 - totalDeformation * 0.05)

    // Tetrahedron vertices (normalized to -1..1, centered)
    // Will be scaled to canvas size
    const baseVertices = [
      { name: 'Climate', x: 0, y: -0.8, z: 0 },
      { name: 'Airflow', x: 0.7, y: 0.4, z: 0 },
      { name: 'Irrigation', x: -0.7, y: 0.4, z: 0 },
      { name: 'Plant', x: 0, y: 0, z: 0.9 },
    ]

    // Apply deformation (asymmetry based on subsystem influence)
    const vertices = baseVertices.map((v, i) => {
      const subsystem = state.subsystems?.[i] || {}
      const contribution = (subsystem.drift_contribution_pct || 0) / 100

      return {
        ...v,
        x: v.x * (1 - totalDeformation * contribution * 0.1),
        y: v.y * (1 - totalDeformation * contribution * 0.1),
        z: v.z * (1 - totalDeformation * contribution * 0.05),
      }
    })

    // Edges connecting vertices
    const edges = [
      [0, 1],
      [0, 2],
      [0, 3],
      [1, 2],
      [1, 3],
      [2, 3],
    ]

    // Project 3D to 2D with perspective
    const project = (v: any) => {
      const scale = 400 / (5 + v.z)
      return {
        x: 500 + v.x * scale,
        y: 350 + v.y * scale,
        z: v.z,
      }
    }

    const projectedVertices = vertices.map(project)

    // Color based on system state
    let edgeColor = '#60a5fa' // blue
    let coreGlow = '#60a5fa'
    if (state.state === 'critical') {
      edgeColor = '#ef4444'
      coreGlow = '#ff6b6b'
    } else if (state.state === 'instability') {
      edgeColor = '#f97316'
      coreGlow = '#fb923c'
    } else if (state.state === 'drift') {
      edgeColor = '#eab308'
      coreGlow = '#facc15'
    }

    // Draw SVG
    const svg = svgRef.current
    svg.innerHTML = `
      <defs>
        <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
          <feMerge>
            <feMergeNode in="coloredBlur"/>
            <feMergeNode in="SourceGraphic"/>
          </feMerge>
        </filter>
        <style>
          @keyframes breathe {
            0%, 100% { r: ${coreRadius * 40}px; }
            50% { r: ${coreRadius * 50}px; }
          }
          @keyframes drift-ambient {
            0%, 100% { transform: translate(0, 0); }
            25% { transform: translate(2px, 1px); }
            50% { transform: translate(0, 2px); }
            75% { transform: translate(-2px, 1px); }
          }
          @keyframes warning-jitter {
            0%, 100% { transform: translate(0, 0); }
            10% { transform: translate(1px, 0); }
            20% { transform: translate(-1px, 0); }
            30% { transform: translate(0.5px, 0.5px); }
            40% { transform: translate(-0.5px, 0.5px); }
            50% { transform: translate(0, -1px); }
            60% { transform: translate(1px, 0); }
            70% { transform: translate(-0.5px, 0); }
            80% { transform: translate(0.5px, 0.5px); }
            90% { transform: translate(-1px, 0); }
          }
          .tetrahedron {
            animation: drift-ambient ${12 - drift * 6}s infinite ease-in-out;
          }
          .core {
            animation: breathe 2s ease-in-out infinite;
            ${state.state === 'critical' || state.state === 'instability' ? 'animation: breathe 1s ease-in-out infinite, warning-jitter 0.6s ease-in-out infinite;' : ''}
          }
        </style>
      </defs>

      <!-- Edges -->
      <g class="tetrahedron" stroke="${edgeColor}" stroke-width="2" fill="none" opacity="0.6">
        ${edges
          .map(([a, b]) => {
            const va = projectedVertices[a]
            const vb = projectedVertices[b]
            return `<line x1="${va.x}" y1="${va.y}" x2="${vb.x}" y2="${vb.y}"/>`
          })
          .join('')}
      </g>

      <!-- Vertices -->
      <g class="tetrahedron" fill="${edgeColor}" opacity="0.8">
        ${projectedVertices
          .map((v) => `<circle cx="${v.x}" cy="${v.y}" r="6"/>`)
          .join('')}
      </g>

      <!-- Core (coherence center) -->
      <g class="tetrahedron" opacity="${Math.max(0.5, 1 - drift * 0.5)}">
        <circle class="core" cx="500" cy="350" fill="${coreGlow}" opacity="0.4" filter="url(#glow)"/>
        <circle cx="500" cy="350" r="3" fill="${coreGlow}" opacity="0.9"/>
      </g>

      <!-- No-action divergence (if projection exists) -->
      ${
        state.no_action_projection && state.no_action_projection.length > 0
          ? `
        <g stroke="#ef4444" stroke-width="1" fill="none" opacity="0.3" stroke-dasharray="4,4">
          <line x1="500" y1="350" x2="${500 + Math.random() * 50 - 25}" y2="${350 + Math.random() * 50 - 25}"/>
        </g>
        `
          : ''
      }
    `
  }, [state])

  return (
    <svg
      ref={svgRef}
      viewBox="0 0 1000 700"
      className="w-full h-full"
      style={{
        filter: 'drop-shadow(0 0 30px rgba(96, 165, 250, 0.2))',
      }}
    />
  )
}
