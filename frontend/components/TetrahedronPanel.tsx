'use client'

import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'

interface TetrahedronPanelProps {
  frame: Record<string, unknown>
}

interface Particle {
  position: THREE.Vector3
  velocity: THREE.Vector3
  life: number
  maxLife: number
  color: THREE.Color
}

interface ConnectionFlow {
  fromIdx: number
  toIdx: number
  progress: number
  intensity: number
}

export default function TetrahedronPanel({ frame }: TetrahedronPanelProps) {
  const drift = Math.max(0, Math.min(1, (frame.structural_drift_score as number) || 0))
  const stability = Math.max(0, Math.min(1, (frame.relational_stability_score as number) || 0))
  const coherence = Math.max(0, Math.min(1, (frame.coherence_score as number) || 0))
  const confidence = Math.max(0, Math.min(1, (frame.confidence as number) || 0))

  const containerRef = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<THREE.Scene | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const tetraRef = useRef<THREE.Group | null>(null)
  const particlesRef = useRef<Particle[]>([])
  const connectionFlowsRef = useRef<ConnectionFlow[]>([])
  const threatRingsRef = useRef<THREE.Mesh[]>([])
  const orbitalRingsRef = useRef<THREE.Mesh[]>([])
  const nestedTetrasRef = useRef<THREE.Group[]>([])
  const mouseRef = useRef({ x: 0, y: 0, down: false })
  const rotationRef = useRef({ x: 0, y: 0 })

  const [avgScore, setAvgScore] = useState(0)
  const [threatLevel, setThreatLevel] = useState(0)
  const [isCritical, setIsCritical] = useState(false)
  const prevValuesRef = useRef({ drift, stability, coherence, confidence })

  const metrics = [
    { label: 'Coherence', value: coherence, color: 0x06b6d4 },
    { label: 'Drift', value: drift, color: 0x3b82f6 },
    { label: 'Stability', value: stability, color: 0x22c55e },
    { label: 'Confidence', value: confidence, color: 0xf97316 },
  ]

  useEffect(() => {
    const avg = (drift + stability + coherence + confidence) / 4
    setAvgScore(avg)

    const threat = Math.max(
      drift >= 0.78 ? 1 : drift >= 0.55 ? 0.7 : 0,
      (1 - stability) >= 0.55 ? 1 : (1 - stability) >= 0.35 ? 0.7 : 0,
    )
    setThreatLevel(threat)
    setIsCritical(threat >= 1)
  }, [drift, stability, coherence, confidence])

  useEffect(() => {
    if (!containerRef.current) return

    // Initialize Three.js scene
    const width = containerRef.current.clientWidth
    const height = containerRef.current.clientHeight

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000)
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })

    renderer.setSize(width, height)
    renderer.setClearColor(0x050710, 0.1)
    containerRef.current.appendChild(renderer.domElement)

    camera.position.z = 5

    sceneRef.current = scene
    cameraRef.current = camera
    rendererRef.current = renderer

    // Animated background
    const canvas = renderer.domElement
    const ctx = (canvas as any).getContext('webgl', { preserveDrawingBuffer: true })

    // Create tetrahedron group
    const tetraGroup = new THREE.Group()
    scene.add(tetraGroup)
    tetraRef.current = tetraGroup

    // Tetrahedron vertices
    const vertices = [
      new THREE.Vector3(0, 2, 0),
      new THREE.Vector3(-1.5, -1, 1.5),
      new THREE.Vector3(1.5, -1, 1.5),
      new THREE.Vector3(0, -1, -1.5),
    ]

    // Create main tetrahedron with wireframe
    const geometry = new THREE.TetrahedronGeometry(1.8, 0)
    const material = new THREE.MeshPhongMaterial({
      color: 0x3b82f6,
      wireframe: false,
      emissive: 0x1e40af,
      emissiveIntensity: 0.3,
      transparent: true,
      opacity: 0.1,
    })
    const mainTetra = new THREE.Mesh(geometry, material)
    tetraGroup.add(mainTetra)

    // Tetrahedron edges
    const edges = [
      [0, 1], [0, 2], [0, 3],
      [1, 2], [1, 3], [2, 3],
    ]

    edges.forEach((edge) => {
      const points = [vertices[edge[0]], vertices[edge[1]]]
      const edgeGeometry = new THREE.BufferGeometry().setFromPoints(points)
      const edgeMaterial = new THREE.LineBasicMaterial({ color: 0x60a5fa, linewidth: 2 })
      const line = new THREE.Line(edgeGeometry, edgeMaterial)
      tetraGroup.add(line)
    })

    // Vertex nodes
    vertices.forEach((vertex, idx) => {
      const nodeGeometry = new THREE.IcosahedronGeometry(0.15, 4)
      const nodeMaterial = new THREE.MeshPhongMaterial({
        color: metrics[idx].color,
        emissive: metrics[idx].color,
        emissiveIntensity: 0.5,
      })
      const node = new THREE.Mesh(nodeGeometry, nodeMaterial)
      node.position.copy(vertex)
      tetraGroup.add(node)

      // Vertex glow
      const glowGeometry = new THREE.IcosahedronGeometry(0.2, 4)
      const glowMaterial = new THREE.MeshBasicMaterial({
        color: metrics[idx].color,
        transparent: true,
        opacity: 0.2,
      })
      const glow = new THREE.Mesh(glowGeometry, glowMaterial)
      glow.position.copy(vertex)
      tetraGroup.add(glow)
    })

    // Orbital rings
    for (let i = 0; i < 3; i++) {
      const ringGeometry = new THREE.TorusGeometry(2.5 + i * 0.5, 0.05, 32, 100)
      const ringMaterial = new THREE.MeshBasicMaterial({
        color: 0x38bdf8,
        transparent: true,
        opacity: 0.15 - i * 0.04,
      })
      const ring = new THREE.Mesh(ringGeometry, ringMaterial)
      ring.rotation.x = Math.PI * 0.3 + i * 0.2
      ring.rotation.y = i * 0.3
      tetraGroup.add(ring)
      orbitalRingsRef.current.push(ring)
    }

    // Threat rings (initially hidden)
    for (let i = 0; i < 3; i++) {
      const threatGeometry = new THREE.TorusGeometry(0.5, 0.05, 32, 100)
      const threatMaterial = new THREE.MeshBasicMaterial({
        color: 0xff4444,
        transparent: true,
        opacity: 0,
      })
      const threatRing = new THREE.Mesh(threatGeometry, threatMaterial)
      threatRing.position.y = -0.5
      tetraGroup.add(threatRing)
      threatRingsRef.current.push(threatRing)
    }

    // Nested smaller tetrahedrons
    for (let i = 0; i < 2; i++) {
      const nestedGeometry = new THREE.TetrahedronGeometry(0.8 - i * 0.3, 0)
      const nestedMaterial = new THREE.MeshPhongMaterial({
        color: 0x06b6d4,
        wireframe: true,
        transparent: true,
        opacity: 0.3 - i * 0.1,
      })
      const nestedTetra = new THREE.Mesh(nestedGeometry, nestedMaterial)
      nestedTetra.rotation.x = (i + 1) * Math.PI * 0.3
      nestedTetra.rotation.y = (i + 1) * Math.PI * 0.2
      const nestedGroup = new THREE.Group()
      nestedGroup.add(nestedTetra)
      tetraGroup.add(nestedGroup)
      nestedTetrasRef.current.push(nestedGroup)
    }

    // Lighting
    const light1 = new THREE.PointLight(0x3b82f6, 1.5)
    light1.position.set(5, 5, 5)
    scene.add(light1)

    const light2 = new THREE.PointLight(0x06b6d4, 1)
    light2.position.set(-5, -5, 5)
    scene.add(light2)

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.3)
    scene.add(ambientLight)

    // Mouse controls
    const onMouseMove = (e: MouseEvent) => {
      if (!containerRef.current) return
      const rect = containerRef.current.getBoundingClientRect()
      mouseRef.current.x = ((e.clientX - rect.left) / width) * 2 - 1
      mouseRef.current.y = -((e.clientY - rect.top) / height) * 2 + 1
    }

    const onMouseDown = () => {
      mouseRef.current.down = true
    }

    const onMouseUp = () => {
      mouseRef.current.down = false
    }

    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mousedown', onMouseDown)
    window.addEventListener('mouseup', onMouseUp)

    // Animation loop
    let animationId: number

    const animate = () => {
      animationId = requestAnimationFrame(animate)

      // Update tetrahedron rotation
      if (tetraRef.current) {
        if (mouseRef.current.down) {
          rotationRef.current.y += mouseRef.current.x * 0.02
          rotationRef.current.x += mouseRef.current.y * 0.02
        } else {
          rotationRef.current.y += 0.0008
          rotationRef.current.x += 0.0005
        }

        tetraRef.current.rotation.x = rotationRef.current.x
        tetraRef.current.rotation.y = rotationRef.current.y
      }

      // Update orbital rings
      orbitalRingsRef.current.forEach((ring, i) => {
        ring.rotation.x += 0.001 + i * 0.0005
        ring.rotation.y += 0.001 + i * 0.0008
      })

      // Update nested tetrahedrons
      nestedTetrasRef.current.forEach((nestedGroup, i) => {
        nestedGroup.rotation.x += 0.003 + i * 0.002
        nestedGroup.rotation.y -= 0.002 - i * 0.001
      })

      // Update threat rings
      threatRingsRef.current.forEach((ring, i) => {
        const baseScale = 0.3 + threatLevel * 0.8
        ring.scale.set(baseScale + i * 0.2, baseScale + i * 0.2, baseScale + i * 0.2)
        ring.rotation.z += 0.02
        const opacity = threatLevel > 0 ? (0.3 + i * 0.2) * threatLevel : 0
        ;(ring.material as THREE.MeshBasicMaterial).opacity = opacity
      })

      // Update particles
      particlesRef.current = particlesRef.current.filter((p) => {
        p.position.add(p.velocity)
        p.velocity.y -= 0.01
        p.life -= 0.02
        return p.life > 0
      })

      // Update connection flows
      connectionFlowsRef.current.forEach((flow) => {
        flow.progress += 0.01
        if (flow.progress > 1) flow.progress = 0
      })

      renderer.render(scene, camera)
    }

    animate()

    // Handle resize
    const handleResize = () => {
      if (!containerRef.current) return
      const newWidth = containerRef.current.clientWidth
      const newHeight = containerRef.current.clientHeight
      camera.aspect = newWidth / newHeight
      camera.updateProjectionMatrix()
      renderer.setSize(newWidth, newHeight)
    }

    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mousedown', onMouseDown)
      window.removeEventListener('mouseup', onMouseUp)
      window.removeEventListener('resize', handleResize)
      cancelAnimationFrame(animationId)
      renderer.dispose()
      containerRef.current?.removeChild(canvas)
    }
  }, [])

  // Emit particles on value change
  useEffect(() => {
    const threshold = 0.03
    const hasChange =
      Math.abs(drift - prevValuesRef.current.drift) > threshold ||
      Math.abs(stability - prevValuesRef.current.stability) > threshold ||
      Math.abs(coherence - prevValuesRef.current.coherence) > threshold ||
      Math.abs(confidence - prevValuesRef.current.confidence) > threshold

    if (hasChange && tetraRef.current) {
      const vertices = tetraRef.current.children.filter((c) => c instanceof THREE.Mesh)

      vertices.forEach((vertex, idx) => {
        if (vertex instanceof THREE.Mesh) {
          const intensity = metrics[idx].value
          const particleCount = Math.ceil(intensity * 20)

          for (let i = 0; i < particleCount; i++) {
            const angle = Math.random() * Math.PI * 2
            const speed = 0.02 + Math.random() * 0.03
            const particle: Particle = {
              position: vertex.position.clone(),
              velocity: new THREE.Vector3(
                Math.cos(angle) * speed,
                Math.random() * 0.02,
                Math.sin(angle) * speed,
              ),
              life: 1,
              maxLife: 1,
              color: new THREE.Color(metrics[idx].color),
            }
            particlesRef.current.push(particle)
          }
        }
      })

      prevValuesRef.current = { drift, stability, coherence, confidence }
    }
  }, [drift, stability, coherence, confidence])

  const getHealthStatus = () => {
    if (isCritical) return 'CRITICAL'
    if (threatLevel >= 0.7) return 'WARNING'
    if (avgScore >= 0.75) return 'EXCELLENT'
    if (avgScore >= 0.5) return 'NOMINAL'
    return 'DEGRADED'
  }

  const getHealthColor = () => {
    if (isCritical) return '#ef4444'
    if (threatLevel >= 0.7) return '#f59e0b'
    if (avgScore >= 0.75) return '#10b981'
    if (avgScore >= 0.5) return '#3b82f6'
    return '#64748b'
  }

  return (
    <div className="panel tetra-panel-ultimate">
      <div className="panel-head">
        <div>
          <span className="eyebrow">Structural State (3D Tetrahedron)</span>
          <span className="panel-subtitle">Real-time 4D system projection • Interactive 3D visualization</span>
        </div>
        <div
          className={`health-badge-ultimate ${isCritical ? 'critical-pulse' : ''}`}
          style={{ borderColor: getHealthColor(), backgroundColor: `${getHealthColor()}20` }}
        >
          <div className="health-indicator-ultimate" style={{ backgroundColor: getHealthColor() }} />
          <div className="health-text-ultimate">
            <div className="health-label-ultimate">System</div>
            <div className="health-status-ultimate" style={{ color: getHealthColor() }}>
              {getHealthStatus()}
            </div>
          </div>
        </div>
      </div>

      <div className="tetra-ultimate-container">
        <div
          ref={containerRef}
          className="tetra-canvas-ultimate"
          style={{
            position: 'relative',
            width: '100%',
            minHeight: '500px',
            borderRadius: '14px',
            border: '1px solid rgba(59, 130, 246, 0.2)',
            background: 'linear-gradient(135deg, rgba(5, 20, 45, 0.9), rgba(2, 10, 30, 0.95))',
            overflow: 'hidden',
          }}
        >
          <div
            style={{
              position: 'absolute',
              top: '10px',
              right: '10px',
              fontSize: '11px',
              color: '#94a3b8',
              pointerEvents: 'none',
              zIndex: 10,
            }}
          >
            💡 Drag to rotate
          </div>
        </div>

        <div className="tetra-info-ultimate">
          <div className="metrics-display-ultimate">
            {metrics.map((metric) => (
              <div key={metric.label} className="metric-ultimate" style={{ borderColor: `#${metric.color.toString(16)}` }}>
                <div className="metric-label-ultimate">{metric.label}</div>
                <div className="metric-value-ultimate" style={{ color: `#${metric.color.toString(16)}` }}>
                  {(metric.value * 100).toFixed(0)}%
                </div>
                <div className="metric-bar-ultimate">
                  <div
                    className="metric-bar-fill-ultimate"
                    style={{
                      width: `${metric.value * 100}%`,
                      backgroundColor: `#${metric.color.toString(16)}`,
                    }}
                  />
                </div>
              </div>
            ))}
          </div>

          <div className="tetra-summary-ultimate">
            <div className="summary-stat-ultimate">
              <span>Average Score</span>
              <span style={{ color: getHealthColor() }}>{(avgScore * 100).toFixed(1)}%</span>
            </div>
            <div className="summary-stat-ultimate">
              <span>Threat Level</span>
              <span style={{ color: getHealthColor() }}>{(threatLevel * 100).toFixed(0)}%</span>
            </div>
            <div className="summary-stat-ultimate">
              <span>Health Status</span>
              <span style={{ color: getHealthColor() }}>{getHealthStatus()}</span>
            </div>
          </div>

          <div className="interaction-hint-ultimate">
            <p>🎯 Interact with the 3D tetrahedron</p>
            <ul>
              <li>Drag to rotate freely</li>
              <li>Watch particles erupt on changes</li>
              <li>Observe threat rings escalate</li>
              <li>See metric coupling flows</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  )
}
