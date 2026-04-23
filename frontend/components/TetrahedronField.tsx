'use client'

import React, { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'

// Each vertex = one pillar of system health
// Position extends outward when healthy, collapses inward when stressed
const VERTEX_DEFS = [
  { key: 'COHERENCE',  label: 'Coherence',  base: new THREE.Vector3(0, 1, 0),           baseColor: '#06b6d4' },
  { key: 'STABILITY',  label: 'Stability',  base: new THREE.Vector3(-0.866, -0.5, 0),   baseColor: '#22c55e' },
  { key: 'DRIFT',      label: 'Drift',      base: new THREE.Vector3(0.866, -0.5, 0),    baseColor: '#f59e0b' },
  { key: 'CONTROL',    label: 'Control',    base: new THREE.Vector3(0, -0.2, 0.8),      baseColor: '#818cf8' },
]

const EDGE_PAIRS: [number, number][] = [
  [0, 1], [1, 2], [2, 0],
  [0, 3], [1, 3], [2, 3],
]

const TRAIL_MAX = 60

interface TetrahedronData {
  interpolatedDrift: number
  interpolatedStability: number
  interpolatedCoherence: number
  interpolatedConfidence: number
}

interface TetrahedronFieldProps {
  data: TetrahedronData
  phaseProgress: number
  phase: string
  escalationLevel: number
  hasThresholdCrossed: boolean
  isApproachingFailure: boolean
  decisionRationale?: string | null
}

interface LabelState {
  x: number
  y: number
  label: string
  score: number
  color: string
}

function scoreColor(s: number): string {
  if (s > 0.65) return '#22c55e'
  if (s > 0.35) return '#eab308'
  return '#ef4444'
}

export function TetrahedronField({
  data,
  phase,
  escalationLevel,
  isApproachingFailure,
  decisionRationale,
}: TetrahedronFieldProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const sceneRef = useRef<THREE.Scene | null>(null)
  const groupRef = useRef<THREE.Group | null>(null)
  const frameRef = useRef<number | null>(null)
  const timeRef = useRef(0)
  const rotYRef = useRef(0)
  const trailPointsRef = useRef<THREE.Vector3[]>([])
  const trailLineRef = useRef<THREE.Line | null>(null)
  const trailColorsRef = useRef<Float32Array | null>(null)
  const centerDotRef = useRef<THREE.Mesh | null>(null)
  const directionArrowRef = useRef<THREE.Mesh | null>(null)
  const vMeshesRef = useRef<THREE.Mesh[]>([])
  const eLinesRef = useRef<THREE.Line[]>([])
  const [labels, setLabels] = useState<LabelState[]>([])

  const liveRef = useRef({ data, phase, escalationLevel, isApproachingFailure })
  useEffect(() => {
    liveRef.current = { data, phase, escalationLevel, isApproachingFailure }
  }, [data, phase, escalationLevel, isApproachingFailure])

  useEffect(() => {
    if (!containerRef.current || rendererRef.current) return
    const container = containerRef.current

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x050607)
    sceneRef.current = scene

    const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100)
    camera.position.set(0, 0.1, 3.0)
    camera.lookAt(0, 0, 0)
    cameraRef.current = camera

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    rendererRef.current = renderer
    container.appendChild(renderer.domElement)

    // Group rotates slowly — vertices stay in local space
    const group = new THREE.Group()
    scene.add(group)
    groupRef.current = group

    // Vertex dots
    const dotGeo = new THREE.SphereGeometry(0.035, 10, 10)
    VERTEX_DEFS.forEach(def => {
      const mat = new THREE.MeshBasicMaterial({ color: def.baseColor })
      const mesh = new THREE.Mesh(dotGeo, mat)
      group.add(mesh)
      vMeshesRef.current.push(mesh)
    })

    // Edges
    EDGE_PAIRS.forEach(() => {
      const geo = new THREE.BufferGeometry()
      geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3))
      const mat = new THREE.LineBasicMaterial({ color: 0x334155, transparent: true, opacity: 0.7 })
      const line = new THREE.Line(geo, mat)
      group.add(line)
      eLinesRef.current.push(line)
    })

    // Center trajectory dot (in world space, not group — so it traces world path)
    const centerGeo = new THREE.SphereGeometry(0.05, 10, 10)
    const centerMat = new THREE.MeshBasicMaterial({ color: 0xffffff })
    const centerDot = new THREE.Mesh(centerGeo, centerMat)
    scene.add(centerDot)
    centerDotRef.current = centerDot

    // Trail line (world space)
    const trailBuf = new Float32Array(TRAIL_MAX * 3)
    const trailGeo = new THREE.BufferGeometry()
    trailGeo.setAttribute('position', new THREE.BufferAttribute(trailBuf, 3))
    const trailColors = new Float32Array(TRAIL_MAX * 3)
    trailColorsRef.current = trailColors
    trailGeo.setAttribute('color', new THREE.BufferAttribute(trailColors, 3))
    const trailMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.55 })
    const trailLine = new THREE.Line(trailGeo, trailMat)
    scene.add(trailLine)
    trailLineRef.current = trailLine

    // Direction arrow (subtle): points along latest movement vector.
    const arrowGeo = new THREE.ConeGeometry(0.04, 0.12, 10)
    arrowGeo.rotateX(Math.PI / 2) // make cone point along +Z
    const arrowMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.35 })
    const arrow = new THREE.Mesh(arrowGeo, arrowMat)
    scene.add(arrow)
    directionArrowRef.current = arrow

    const resize = () => {
      if (!container || !cameraRef.current || !rendererRef.current) return
      const w = container.clientWidth || window.innerWidth
      const h = container.clientHeight || window.innerHeight
      cameraRef.current.aspect = w / Math.max(h, 1)
      cameraRef.current.updateProjectionMatrix()
      rendererRef.current.setSize(w, h, false)
    }

    const animate = () => {
      frameRef.current = requestAnimationFrame(animate)
      timeRef.current += 0.016

      const { data: d, escalationLevel: esc, isApproachingFailure: failing } = liveRef.current

      // Health scores per vertex: 0 = critical, 1 = perfect
      const scores = [
        d.interpolatedCoherence,
        d.interpolatedStability,
        Math.max(0, 1 - d.interpolatedDrift),
        Math.max(0, 1 - esc / 3),
      ]

      // Slow rotation to show depth
      rotYRef.current += 0.004
      if (groupRef.current) groupRef.current.rotation.y = rotYRef.current

      // Deform each vertex outward (healthy) or inward (stressed)
      VERTEX_DEFS.forEach((def, i) => {
        const score = scores[i]
        // Healthy = extends to 1.3x, stressed = collapses to 0.55x
        const radius = 0.55 + score * 0.75
        const jitter = failing ? (Math.random() - 0.5) * 0.025 : 0
        const target = def.base.clone().normalize().multiplyScalar(radius)
        target.x += jitter
        target.y += jitter * 0.5

        vMeshesRef.current[i].position.copy(target)
        ;(vMeshesRef.current[i].material as THREE.MeshBasicMaterial).color.setStyle(scoreColor(score))
      })

      // Update edges — color shows tension between connected vertices
      EDGE_PAIRS.forEach(([a, b], i) => {
        const pa = vMeshesRef.current[a].position
        const pb = vMeshesRef.current[b].position
        const arr = eLinesRef.current[i].geometry.attributes.position.array as Float32Array
        arr[0] = pa.x; arr[1] = pa.y; arr[2] = pa.z
        arr[3] = pb.x; arr[4] = pb.y; arr[5] = pb.z
        eLinesRef.current[i].geometry.attributes.position.needsUpdate = true
        const edgeScore = Math.min(scores[a], scores[b])
        ;(eLinesRef.current[i].material as THREE.LineBasicMaterial).color.setStyle(scoreColor(edgeScore))
        ;(eLinesRef.current[i].material as THREE.LineBasicMaterial).opacity = 0.35 + edgeScore * 0.55
      })

      // Center of mass in world space
      const localCenter = new THREE.Vector3()
      vMeshesRef.current.forEach(m => localCenter.add(m.position))
      localCenter.divideScalar(4)
      const worldCenter = localCenter.clone().applyMatrix4(groupRef.current!.matrixWorld)

      if (centerDotRef.current) {
        centerDotRef.current.position.copy(worldCenter)
        ;(centerDotRef.current.material as THREE.MeshBasicMaterial).color.setStyle(
          scoreColor(scores.reduce((a, b) => a + b, 0) / 4)
        )
      }

      // Trail
      trailPointsRef.current.push(worldCenter.clone())
      if (trailPointsRef.current.length > TRAIL_MAX) trailPointsRef.current.shift()
      if (trailLineRef.current) {
        const trail = trailPointsRef.current
        const arr = trailLineRef.current.geometry.attributes.position.array as Float32Array
        const carr = trailLineRef.current.geometry.attributes.color.array as Float32Array
        const n = Math.max(1, trail.length)
        trail.forEach((p, i) => {
          arr[i * 3] = p.x; arr[i * 3 + 1] = p.y; arr[i * 3 + 2] = p.z
          // Fade tail: oldest is dim, newest is bright.
          const t = n <= 1 ? 1 : i / (n - 1)
          const k = 0.14 + 0.86 * t
          carr[i * 3] = k
          carr[i * 3 + 1] = k
          carr[i * 3 + 2] = k
        })
        trailLineRef.current.geometry.attributes.position.needsUpdate = true
        trailLineRef.current.geometry.attributes.color.needsUpdate = true
        trailLineRef.current.geometry.setDrawRange(0, trail.length)
      }

      // Direction arrow: align +Z to movement vector.
      if (directionArrowRef.current) {
        const trail = trailPointsRef.current
        if (trail.length >= 2) {
          const last = trail[trail.length - 1]
          const prev = trail[trail.length - 2]
          const dir = last.clone().sub(prev)
          if (dir.lengthSq() > 1e-8) {
            dir.normalize()
            directionArrowRef.current.position.copy(last)
            const from = new THREE.Vector3(0, 0, 1)
            const q = new THREE.Quaternion().setFromUnitVectors(from, dir)
            directionArrowRef.current.quaternion.copy(q)
            ;(directionArrowRef.current.material as THREE.MeshBasicMaterial).opacity = 0.22 + 0.18 * Math.min(1, Math.max(0, (scores[0] + scores[1] + scores[2] + scores[3]) / 4))
          }
        }
      }

      renderer.render(scene, camera)

      // CSS labels — project world positions
      if (cameraRef.current && container) {
        const w = container.clientWidth
        const h = container.clientHeight
        const newLabels = VERTEX_DEFS.map((def, i) => {
          const wp = vMeshesRef.current[i].getWorldPosition(new THREE.Vector3())
          wp.project(cameraRef.current!)
          return {
            x: (wp.x * 0.5 + 0.5) * w,
            y: (-wp.y * 0.5 + 0.5) * h,
            label: def.label,
            score: scores[i],
            color: scoreColor(scores[i]),
          }
        })
        setLabels(newLabels)
      }
    }

    resize()
    animate()
    const ro = new ResizeObserver(resize)
    ro.observe(container)
    window.addEventListener('resize', resize)

    return () => {
      window.removeEventListener('resize', resize)
      ro.disconnect()
      if (frameRef.current) cancelAnimationFrame(frameRef.current)
      scene.traverse((o: any) => {
        o.geometry?.dispose()
        Array.isArray(o.material) ? o.material.forEach((m: THREE.Material) => m.dispose()) : o.material?.dispose()
      })
      renderer.dispose()
      renderer.forceContextLoss()
      renderer.domElement.parentNode?.removeChild(renderer.domElement)
      vMeshesRef.current = []
      eLinesRef.current = []
      rendererRef.current = null
      cameraRef.current = null
      sceneRef.current = null
    }
  }, [])

  const atmosphereColor = phase === 'Critical' ? '#ef4444'
    : phase === 'Instability forming' ? '#f97316'
    : phase === 'Drift forming' ? '#eab308'
    : '#22c55e'

  return (
    <div ref={containerRef} style={{ width: '100%', height: '100%', position: 'relative' }}>
      {/* Phase atmosphere: radial glow that shifts with system state */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          background: `radial-gradient(ellipse 60% 50% at 50% 50%, ${atmosphereColor}0a 0%, transparent 70%)`,
          pointerEvents: 'none',
          zIndex: 1,
          transition: 'background 1.8s ease',
        }}
      />

      {/* Axis scale labels */}
      <div style={{
        position: 'absolute',
        bottom: 10,
        left: 10,
        zIndex: 3,
        pointerEvents: 'none',
        display: 'flex',
        flexDirection: 'column',
        gap: 4,
      }}>
        <div style={{ fontSize: 9, color: '#64748b', letterSpacing: '0.6px', textTransform: 'uppercase', fontWeight: 800 }}>
          Scale: 0–100
        </div>
        <div style={{ fontSize: 9, color: '#475569', fontWeight: 700 }}>
          Center = system centroid
        </div>
      </div>

      {/* Decision rationale overlay */}
      {decisionRationale && (
        <div style={{
          position: 'absolute',
          bottom: 10,
          right: 10,
          zIndex: 3,
          pointerEvents: 'none',
          maxWidth: 220,
          background: 'rgba(5, 6, 7, 0.72)',
          border: '1px solid rgba(255,255,255,0.06)',
          borderRadius: 10,
          padding: '10px 12px',
          backdropFilter: 'blur(6px)',
        }}>
          <div style={{
            fontSize: 9,
            color: '#94a3b8',
            letterSpacing: '0.8px',
            textTransform: 'uppercase',
            fontWeight: 950,
            marginBottom: 4,
          }}>
            Why this shape
          </div>
          <div style={{
            fontSize: 11,
            color: '#e2e8f0',
            lineHeight: 1.45,
            fontWeight: 600,
          }}>
            {decisionRationale}
          </div>
        </div>
      )}

      {labels.map((lbl, i) => (
        <div
          key={i}
          style={{
            position: 'absolute',
            left: lbl.x,
            top: lbl.y,
            transform: 'translate(-50%, -220%)',
            pointerEvents: 'none',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            gap: 2,
            zIndex: 2,
          }}
        >
          <div style={{
            fontSize: 10,
            color: lbl.color,
            fontWeight: 700,
            letterSpacing: '0.8px',
            textTransform: 'uppercase',
            whiteSpace: 'nowrap',
            fontFamily: 'system-ui, sans-serif',
          }}>
            {lbl.label}
          </div>
          <div style={{
            fontSize: 11,
            color: lbl.color,
            fontVariantNumeric: 'tabular-nums',
            fontFamily: 'system-ui, sans-serif',
            opacity: 0.85,
          }}>
            {Math.round(lbl.score * 100)}%
          </div>
        </div>
      ))}
    </div>
  )
}
