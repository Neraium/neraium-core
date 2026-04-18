'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { TetrahedronState, Point3D } from '@/lib/decisionToUI'

interface EnhancedTetrahedronVizProps {
  tetrahedronState: TetrahedronState
  isInteractive?: boolean
}

const VERTICES = {
  STRUCTURAL: { pos: [1.0, 1.0, 1.0], color: 0x06b6d4 },
  RELATIONAL: { pos: [1.0, -1.0, -1.0], color: 0x22c55e },
  TEMPORAL: { pos: [-1.0, -1.0, 1.0], color: 0xf59e0b },
  AUTHORITY: { pos: [-1.0, 1.0, -1.0], color: 0x3b82f6 },
}

const SCALE = 2.0

const severityColor = (severity: number): number => {
  if (severity >= 0.9) return 0xef4444
  if (severity >= 0.75) return 0xf97316
  if (severity >= 0.5) return 0xeab308
  return 0x22c55e
}

const toCoord = (p: Point3D): THREE.Vector3 => {
  const v1 = new THREE.Vector3(...(VERTICES.STRUCTURAL.pos as [number, number, number]))
  const v2 = new THREE.Vector3(...(VERTICES.RELATIONAL.pos as [number, number, number]))
  const v3 = new THREE.Vector3(...(VERTICES.TEMPORAL.pos as [number, number, number]))
  const v4 = new THREE.Vector3(...(VERTICES.AUTHORITY.pos as [number, number, number]))

  const result = new THREE.Vector3()
  result.addScaledVector(v1, p.x)
  result.addScaledVector(v2, p.y)
  result.addScaledVector(v3, p.z)
  result.addScaledVector(v4, Math.max(0, 1 - (p.x + p.y + p.z) / 3))
  return result.multiplyScalar(SCALE)
}

export default function EnhancedTetrahedronViz({ tetrahedronState, isInteractive = true }: EnhancedTetrahedronVizProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const groupRef = useRef<THREE.Group | null>(null)
  const trailGroupRef = useRef<THREE.Group | null>(null)
  const statePointRef = useRef<THREE.Mesh | null>(null)
  const stateGlowRef = useRef<THREE.Mesh | null>(null)
  const stateHaloRef = useRef<THREE.Mesh | null>(null)
  const targetPositionRef = useRef<THREE.Vector3>(toCoord(tetrahedronState.currentPosition))
  const targetSeverityRef = useRef<number>(tetrahedronState.severityScalar)
  const rotationRef = useRef({ x: 0.2, y: 0.32 })
  const [hasError, setHasError] = useState(false)

  const vignette = useMemo(() => {
    const edge = 0.18 + tetrahedronState.severityScalar * 0.3
    return `radial-gradient(circle at 50% 38%, rgba(15, 23, 42, 0.02) 31%, rgba(2, 6, 23, ${edge}) 97%)`
  }, [tetrahedronState.severityScalar])

  useEffect(() => {
    if (!containerRef.current) return
    const container = containerRef.current

    let width = container.clientWidth
    let height = container.clientHeight

    // Guard: container must have real dimensions
    if (width === 0 || height === 0) {
      const rect = container.getBoundingClientRect()
      width = rect.width || 640
      height = rect.height || 640
    }

    try {
      const scene = new THREE.Scene()
      scene.fog = new THREE.FogExp2(0x020617, 0.09)

      const camera = new THREE.PerspectiveCamera(66, width / height, 0.1, 1000)
      camera.position.set(0, 1, 5.5)

      const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
      renderer.setSize(width, height)
      renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
      renderer.setClearColor(0x020617, 0.04)
      renderer.domElement.style.width = '100%'
      renderer.domElement.style.height = '100%'
      renderer.domElement.style.display = 'block'
      container.appendChild(renderer.domElement)

      const group = new THREE.Group()
      scene.add(group)

      const vertices = Object.values(VERTICES)
      ;[[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]].forEach(([a, b]) => {
        const geom = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(...(vertices[a].pos as [number, number, number])).multiplyScalar(SCALE),
          new THREE.Vector3(...(vertices[b].pos as [number, number, number])).multiplyScalar(SCALE),
        ])
        const mat = new THREE.LineBasicMaterial({ color: 0x475569, transparent: true, opacity: 0.55 })
        group.add(new THREE.Line(geom, mat))
      })

      Object.values(VERTICES).forEach((vertex) => {
        const pos = new THREE.Vector3(...(vertex.pos as [number, number, number])).multiplyScalar(SCALE)
        const node = new THREE.Mesh(
          new THREE.IcosahedronGeometry(0.22, 2),
          new THREE.MeshStandardMaterial({ color: vertex.color, emissive: vertex.color, emissiveIntensity: 0.18 }),
        )
        node.position.copy(pos)
        group.add(node)
      })

      const trailGroup = new THREE.Group()
      group.add(trailGroup)

      const point = new THREE.Mesh(
        new THREE.SphereGeometry(0.36, 24, 24),
        new THREE.MeshPhongMaterial({ color: 0x22c55e, emissive: 0x22c55e, emissiveIntensity: 1.0, shininess: 150 }),
      )
      point.position.copy(targetPositionRef.current)
      group.add(point)

      const glow = new THREE.Mesh(
        new THREE.SphereGeometry(0.72, 24, 24),
        new THREE.MeshBasicMaterial({ color: 0x22c55e, transparent: true, opacity: 0.38 }),
      )
      glow.position.copy(targetPositionRef.current)
      group.add(glow)

      const halo = new THREE.Mesh(
        new THREE.SphereGeometry(1.1, 24, 24),
        new THREE.MeshBasicMaterial({ color: 0x22c55e, transparent: true, opacity: 0.08 }),
      )
      halo.position.copy(targetPositionRef.current)
      group.add(halo)

      const key = new THREE.PointLight(0x93c5fd, 1.25)
      key.position.set(6, 5, 6)
      scene.add(key)

      const fill = new THREE.PointLight(0x22d3ee, 0.88)
      fill.position.set(-6, -4, 5)
      scene.add(fill)

      scene.add(new THREE.AmbientLight(0xe2e8f0, 0.78))

      cameraRef.current = camera
      rendererRef.current = renderer
      groupRef.current = group
      trailGroupRef.current = trailGroup
      statePointRef.current = point
      stateGlowRef.current = glow
      stateHaloRef.current = halo

      let animationId = 0
      const animate = () => {
        animationId = requestAnimationFrame(animate)

        if (groupRef.current) {
          rotationRef.current.y += isInteractive ? 0.000095 : 0.000061
          groupRef.current.rotation.x = rotationRef.current.x
          groupRef.current.rotation.y = rotationRef.current.y
        }

        if (statePointRef.current && stateGlowRef.current && stateHaloRef.current) {
          const target = targetPositionRef.current
          statePointRef.current.position.lerp(target, 0.038)
          stateGlowRef.current.position.lerp(target, 0.038)
          stateHaloRef.current.position.lerp(target, 0.038)

          const sev = targetSeverityRef.current
          const color = severityColor(sev)
          const pointMat = statePointRef.current.material as THREE.MeshPhongMaterial
          pointMat.color.setHex(color)
          pointMat.emissive.setHex(color)
          pointMat.emissiveIntensity = 0.8 + sev * 0.58

          const glowMat = stateGlowRef.current.material as THREE.MeshBasicMaterial
          glowMat.color.setHex(color)
          glowMat.opacity = 0.28 + sev * 0.34

          const haloMat = stateHaloRef.current.material as THREE.MeshBasicMaterial
          haloMat.color.setHex(color)
          haloMat.opacity = 0.06 + sev * 0.14

          const t = performance.now() * 0.0009
          const pulse = 1 + Math.sin(t) * (0.04 + sev * 0.06)
          stateGlowRef.current.scale.setScalar(pulse)
          stateHaloRef.current.scale.setScalar(1 + Math.sin(t * 0.7) * (0.02 + sev * 0.03))
        }

        renderer.render(scene, camera)
      }

      animate()

      const onResize = () => {
        if (!containerRef.current || !cameraRef.current || !rendererRef.current) return
        const w = containerRef.current.clientWidth || 640
        const h = containerRef.current.clientHeight || 640
        cameraRef.current.aspect = w / h
        cameraRef.current.updateProjectionMatrix()
        rendererRef.current.setSize(w, h)
      }
      window.addEventListener('resize', onResize)

      return () => {
        window.removeEventListener('resize', onResize)
        cancelAnimationFrame(animationId)
        renderer.dispose()
        if (container.contains(renderer.domElement)) container.removeChild(renderer.domElement)
      }
    } catch (e) {
      setHasError(true)
    }
  }, [isInteractive])

  useEffect(() => {
    targetPositionRef.current = toCoord(tetrahedronState.currentPosition)
    targetSeverityRef.current = tetrahedronState.severityScalar

    const trailGroup = trailGroupRef.current
    if (!trailGroup) return
    while (trailGroup.children.length) trailGroup.remove(trailGroup.children[0])

    const points = tetrahedronState.trailPoints.map((tp) => toCoord(tp.position))
    for (let i = 1; i < points.length; i++) {
      const p = i / points.length
      const color = severityColor(tetrahedronState.severityScalar)
      const segGeom = new THREE.BufferGeometry().setFromPoints([points[i - 1], points[i]])
      const segMat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.08 + p * 0.38 })
      trailGroup.add(new THREE.Line(segGeom, segMat))

      const node = new THREE.Mesh(
        new THREE.SphereGeometry(0.055 + p * 0.07, 10, 10),
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.12 + p * 0.40 }),
      )
      node.position.copy(points[i])
      trailGroup.add(node)
    }
  }, [tetrahedronState])

  return (
    <div
      style={{
        width: '100%',
        height: '100%',
        minHeight: '500px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        position: 'relative',
      }}
    >
      {hasError ? (
        <div
          style={{
            width: '100%',
            height: '100%',
            minHeight: '500px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            borderRadius: '24px',
            border: '1px dashed rgba(148,163,184,0.25)',
            background: 'rgba(15,23,42,0.4)',
            color: 'rgba(148,163,184,0.7)',
            fontSize: '14px',
          }}
        >
          System visual unavailable
        </div>
      ) : (
        <div
          ref={containerRef}
          style={{
            width: '100%',
            height: '100%',
            minHeight: '500px',
            borderRadius: '24px',
            border: '1px solid rgba(30,41,59,0.5)',
            background: vignette,
            opacity: 1,
            zIndex: 1,
            overflow: 'hidden',
          }}
        />
      )}
    </div>
  )
}
