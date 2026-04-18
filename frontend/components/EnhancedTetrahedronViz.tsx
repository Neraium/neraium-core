'use client'

import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { TetrahedronState, Point3D } from '@/lib/decisionToUI'

interface EnhancedTetrahedronVizProps {
  tetrahedronState: TetrahedronState
  isInteractive?: boolean
}

/**
 * Enhanced 3D tetrahedron visualization mapping decision state to 3D space.
 *
 * Core features:
 * - System state as point inside tetrahedron
 * - Trail showing historical trajectory (last 10-20 frames)
 * - Directional arrow showing current velocity
 * - Color coding by severity (green→yellow→orange→red)
 * - Vertex labels showing structural dimensions
 */
export default function EnhancedTetrahedronViz({
  tetrahedronState,
  isInteractive = true,
}: EnhancedTetrahedronVizProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<THREE.Scene | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const tetraGroupRef = useRef<THREE.Group | null>(null)
  const trailGroupRef = useRef<THREE.Group | null>(null)
  const statePointRef = useRef<THREE.Mesh | null>(null)
  const arrowRef = useRef<THREE.Group | null>(null)
  const mouseRef = useRef({ x: 0, y: 0, down: false })
  const rotationRef = useRef({ x: 0.3, y: 0.5 })

  const [hoveredVertex, setHoveredVertex] = useState<string | null>(null)

  // Tetrahedron vertices mapping to structural dimensions
  const VERTICES = {
    STRUCTURAL: { pos: [1.0, 1.0, 1.0], label: 'STRUCTURAL\n(Coherence)', color: 0x06b6d4 },
    RELATIONAL: { pos: [1.0, -1.0, -1.0], label: 'RELATIONAL\n(Stability)', color: 0x22c55e },
    TEMPORAL: { pos: [-1.0, -1.0, 1.0], label: 'TEMPORAL\n(Persistence)', color: 0xf59e0b },
    AUTHORITY: { pos: [-1.0, 1.0, -1.0], label: 'AUTHORITY\n(Confidence)', color: 0x3b82f6 },
  }

  // Map severity to color
  const getSeverityColor = (severity: number): number => {
    if (severity >= 0.9) return 0xef4444 // RED
    if (severity >= 0.75) return 0xf97316 // ORANGE
    if (severity >= 0.5) return 0xeab308 // YELLOW
    return 0x22c55e // GREEN
  }

  // Convert 3D point in [0,1] space to tetrahedron coordinates
  const pointToTetrahedronCoords = (p: Point3D): THREE.Vector3 => {
    const v1 = new THREE.Vector3(VERTICES.STRUCTURAL.pos[0], VERTICES.STRUCTURAL.pos[1], VERTICES.STRUCTURAL.pos[2])
    const v2 = new THREE.Vector3(VERTICES.RELATIONAL.pos[0], VERTICES.RELATIONAL.pos[1], VERTICES.RELATIONAL.pos[2])
    const v3 = new THREE.Vector3(VERTICES.TEMPORAL.pos[0], VERTICES.TEMPORAL.pos[1], VERTICES.TEMPORAL.pos[2])
    const v4 = new THREE.Vector3(VERTICES.AUTHORITY.pos[0], VERTICES.AUTHORITY.pos[1], VERTICES.AUTHORITY.pos[2])

    // Barycentric coordinates: interpolate inside tetrahedron
    // Point at center (0.5, 0.5, 0.5) maps to tetrahedron center
    const t1 = p.x
    const t2 = p.y
    const t3 = p.z
    const t4 = Math.max(0, 1 - (t1 + t2 + t3) / 3)

    const result = new THREE.Vector3()
    result.addScaledVector(v1, t1)
    result.addScaledVector(v2, t2)
    result.addScaledVector(v3, t3)
    result.addScaledVector(v4, t4)

    return result
  }

  // Create directional arrow
  const createArrow = (from: THREE.Vector3, direction: THREE.Vector3): THREE.Group => {
    const arrowGroup = new THREE.Group()

    // Normalize direction
    const dir = direction.clone().normalize()
    const length = Math.min(1.5, direction.length() * 5)

    if (length > 0.1) {
      // Arrow shaft
      const shaftGeom = new THREE.CylinderGeometry(0.05, 0.05, length, 8)
      const shaftMat = new THREE.MeshStandardMaterial({
        color: 0xfbbf24,
        emissive: 0xfbbf24,
        emissiveIntensity: 0.5,
      })
      const shaft = new THREE.Mesh(shaftGeom, shaftMat)
      shaft.position.copy(from)
      shaft.position.addScaledVector(dir, length / 2)
      shaft.lookAt(from.clone().add(dir))
      arrowGroup.add(shaft)

      // Arrow head
      const headGeom = new THREE.ConeGeometry(0.12, 0.3, 8)
      const headMat = new THREE.MeshStandardMaterial({
        color: 0xfbbf24,
        emissive: 0xfbbf24,
        emissiveIntensity: 0.8,
      })
      const head = new THREE.Mesh(headGeom, headMat)
      head.position.copy(from)
      head.position.addScaledVector(dir, length + 0.15)
      head.lookAt(from.clone().add(dir))
      arrowGroup.add(head)
    }

    return arrowGroup
  }

  useEffect(() => {
    if (!containerRef.current) return

    const width = containerRef.current.clientWidth
    const height = containerRef.current.clientHeight

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000)
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })

    renderer.setSize(width, height)
    renderer.setClearColor(0x0f172a, 0.1)
    renderer.shadowMap.enabled = true
    containerRef.current.appendChild(renderer.domElement)

    camera.position.z = 5
    camera.position.y = 1

    sceneRef.current = scene
    cameraRef.current = camera
    rendererRef.current = renderer

    // Create tetrahedron group
    const tetraGroup = new THREE.Group()
    scene.add(tetraGroup)
    tetraGroupRef.current = tetraGroup

    // Draw tetrahedron edges
    const edgeIndices = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    const vertexList = Object.values(VERTICES)

    edgeIndices.forEach((edge) => {
      const v1 = new THREE.Vector3(...(vertexList[edge[0]].pos as [number, number, number]))
      const v2 = new THREE.Vector3(...(vertexList[edge[1]].pos as [number, number, number]))

      const edgeGeom = new THREE.BufferGeometry().setFromPoints([v1, v2])
      const edgeMat = new THREE.LineBasicMaterial({ color: 0x64748b, linewidth: 2 })
      const line = new THREE.Line(edgeGeom, edgeMat)
      tetraGroup.add(line)
    })

    // Vertex spheres with labels
    Object.entries(VERTICES).forEach((entry) => {
      const [name, vertex] = entry as [string, (typeof VERTICES)['STRUCTURAL']]
      const pos = new THREE.Vector3(...(vertex.pos as [number, number, number]))

      // Vertex node
      const nodeGeom = new THREE.IcosahedronGeometry(0.15, 4)
      const nodeMat = new THREE.MeshPhongMaterial({
        color: vertex.color,
        emissive: vertex.color,
        emissiveIntensity: 0.3,
      })
      const node = new THREE.Mesh(nodeGeom, nodeMat)
      node.position.copy(pos)
      node.castShadow = true
      tetraGroup.add(node)

      // Vertex glow
      const glowGeom = new THREE.IcosahedronGeometry(0.22, 4)
      const glowMat = new THREE.MeshBasicMaterial({
        color: vertex.color,
        transparent: true,
        opacity: 0.2,
      })
      const glow = new THREE.Mesh(glowGeom, glowMat)
      glow.position.copy(pos)
      tetraGroup.add(glow)

      // Vertex label
      const canvas = document.createElement('canvas')
      canvas.width = 256
      canvas.height = 256
      const ctx = canvas.getContext('2d')
      if (ctx) {
        ctx.fillStyle = '#e2e8f0'
        ctx.font = 'bold 32px monospace'
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        vertex.label.split('\n').forEach((line, i) => {
          ctx.fillText(line, 128, 100 + i * 50)
        })
      }

      const texture = new THREE.CanvasTexture(canvas)
      const labelGeom = new THREE.PlaneGeometry(1.2, 1.2)
      const labelMat = new THREE.MeshBasicMaterial({ map: texture, transparent: true })
      const label = new THREE.Mesh(labelGeom, labelMat)

      const labelDist = 1.6
      label.position.copy(pos).normalize().multiplyScalar(labelDist)
      label.lookAt(camera.position)
      tetraGroup.add(label)
    })

    // Trail group
    const trailGroup = new THREE.Group()
    tetraGroup.add(trailGroup)
    trailGroupRef.current = trailGroup

    // Draw trail
    if (tetrahedronState.trailPoints.length > 1) {
      const trailPoints = tetrahedronState.trailPoints.map((tp) => pointToTetrahedronCoords(tp.position))

      // Trail line segments with gradient
      for (let i = 1; i < trailPoints.length; i++) {
        const progress = i / trailPoints.length
        const color = getSeverityColor(tetrahedronState.severityScalar)

        const segGeom = new THREE.BufferGeometry().setFromPoints([trailPoints[i - 1], trailPoints[i]])
        const segMat = new THREE.LineBasicMaterial({
          color,
          linewidth: 1 + progress * 2,
          transparent: true,
          opacity: 0.3 + progress * 0.7,
        })
        const line = new THREE.Line(segGeom, segMat)
        trailGroup.add(line)
      }

      // Trail points (decreasing size/opacity toward start)
      trailPoints.forEach((p, idx) => {
        const progress = idx / trailPoints.length
        const pointGeom = new THREE.SphereGeometry(0.08 + progress * 0.08, 8, 8)
        const pointMat = new THREE.MeshBasicMaterial({
          color: getSeverityColor(tetrahedronState.severityScalar),
          transparent: true,
          opacity: 0.2 + progress * 0.6,
        })
        const point = new THREE.Mesh(pointGeom, pointMat)
        point.position.copy(p)
        trailGroup.add(point)
      })
    }

    // Current state point
    const statePos = pointToTetrahedronCoords(tetrahedronState.currentPosition)
    const stateGeom = new THREE.SphereGeometry(0.2, 16, 16)
    const stateMat = new THREE.MeshPhongMaterial({
      color: getSeverityColor(tetrahedronState.severityScalar),
      emissive: getSeverityColor(tetrahedronState.severityScalar),
      emissiveIntensity: 0.8,
      shininess: 100,
    })
    const statePoint = new THREE.Mesh(stateGeom, stateMat)
    statePoint.position.copy(statePos)
    statePoint.castShadow = true
    tetraGroup.add(statePoint)
    statePointRef.current = statePoint

    // Directional arrow
    const velMagnitude = Math.sqrt(
      tetrahedronState.velocity[0] ** 2 +
        tetrahedronState.velocity[1] ** 2 +
        tetrahedronState.velocity[2] ** 2
    )
    if (velMagnitude > 0.01) {
      const velocityDir = new THREE.Vector3(
        tetrahedronState.velocity[0],
        tetrahedronState.velocity[1],
        tetrahedronState.velocity[2]
      ).normalize()

      const arrow = createArrow(statePos, velocityDir)
      tetraGroup.add(arrow)
      arrowRef.current = arrow
    }

    // Lighting
    const light1 = new THREE.PointLight(0x3b82f6, 1.5)
    light1.position.set(5, 5, 5)
    light1.castShadow = true
    scene.add(light1)

    const light2 = new THREE.PointLight(0x06b6d4, 1)
    light2.position.set(-5, -5, 5)
    scene.add(light2)

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4)
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

      // Smooth rotation
      if (tetraGroupRef.current) {
        if (mouseRef.current.down && isInteractive) {
          rotationRef.current.y += mouseRef.current.x * 0.02
          rotationRef.current.x += mouseRef.current.y * 0.02
        } else if (isInteractive) {
          rotationRef.current.y += 0.0005
        }

        // Clamp X rotation to avoid flipping
        rotationRef.current.x = Math.max(-Math.PI / 2.5, Math.min(Math.PI / 2.5, rotationRef.current.x))

        tetraGroupRef.current.rotation.x = rotationRef.current.x
        tetraGroupRef.current.rotation.y = rotationRef.current.y
      }

      // Pulse state point
      if (statePointRef.current) {
        const pulse = 0.18 + Math.sin(Date.now() * 0.005) * 0.05
        statePointRef.current.scale.set(pulse, pulse, pulse)
      }

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
      if (containerRef.current?.contains(renderer.domElement)) {
        containerRef.current.removeChild(renderer.domElement)
      }
    }
  }, [tetrahedronState, isInteractive])

  return (
    <div className="relative w-full">
      <div
        ref={containerRef}
        className="w-full rounded-lg border border-slate-700 bg-gradient-to-br from-slate-900 via-slate-950 to-slate-900"
        style={{ height: '600px' }}
      />
      <div className="absolute bottom-4 right-4 text-xs text-slate-500 pointer-events-none">
        {isInteractive && '💡 Drag to rotate'}
      </div>
      {hoveredVertex && (
        <div className="absolute top-4 left-4 text-sm font-semibold text-slate-300">
          Nearest: {hoveredVertex}
        </div>
      )}
    </div>
  )
}
