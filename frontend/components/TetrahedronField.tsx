'use client'

import React, { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { getPhaseColor } from '@/lib/phaseController'

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
}

export function TetrahedronField({
  data,
  phaseProgress,
  phase,
  escalationLevel,
  hasThresholdCrossed,
  isApproachingFailure,
}: TetrahedronFieldProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<THREE.Scene | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const tetrahedronRef = useRef<THREE.Group | null>(null)
  const edgesRef = useRef<THREE.Line[]>([])
  const verticesRef = useRef<THREE.Mesh[]>([])
  const frameRef = useRef<number | null>(null)
  const timeRef = useRef(0)

  const propsRef = useRef({
    data,
    phaseProgress,
    phase,
    escalationLevel,
    hasThresholdCrossed,
    isApproachingFailure,
  })

  useEffect(() => {
    propsRef.current = {
      data,
      phaseProgress,
      phase,
      escalationLevel,
      hasThresholdCrossed,
      isApproachingFailure,
    }
  }, [data, phaseProgress, phase, escalationLevel, hasThresholdCrossed, isApproachingFailure])

  useEffect(() => {
    if (!containerRef.current || rendererRef.current) return

    const container = containerRef.current

    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x0a0e1a)
    sceneRef.current = scene

    const camera = new THREE.PerspectiveCamera(75, 1, 0.1, 1000)
    camera.position.z = 6.0
    cameraRef.current = camera

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance',
    })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    rendererRef.current = renderer
    container.appendChild(renderer.domElement)

    const tetrahedron = new THREE.Group()
    tetrahedronRef.current = tetrahedron
    scene.add(tetrahedron)

    const baseScale = 1.35
    const baseVertices = [
      new THREE.Vector3(0, 1, 0),
      new THREE.Vector3(-0.866, -0.5, 0),
      new THREE.Vector3(0.866, -0.5, 0),
      new THREE.Vector3(0, -0.2, 0.8),
    ]

    const sphereGeometry = new THREE.SphereGeometry(0.12, 16, 16)

    baseVertices.forEach((vertex) => {
      const material = new THREE.MeshPhongMaterial({
        color: 0x38bdf8,
        emissive: 0x38bdf8,
        emissiveIntensity: 0.3,
      })
      const sphere = new THREE.Mesh(sphereGeometry, material)
      sphere.position.copy(vertex.clone().multiplyScalar(baseScale))
      sphere.userData.basePosition = sphere.position.clone()
      tetrahedron.add(sphere)
      verticesRef.current.push(sphere)
    })

    const edgePairs = [
      [0, 1],
      [1, 2],
      [2, 0],
      [0, 3],
      [1, 3],
      [2, 3],
    ]

    edgePairs.forEach(([startIdx, endIdx]) => {
      const geometry = new THREE.BufferGeometry()
      const start = baseVertices[startIdx].clone().multiplyScalar(baseScale)
      const end = baseVertices[endIdx].clone().multiplyScalar(baseScale)

      geometry.setAttribute(
        'position',
        new THREE.BufferAttribute(
          new Float32Array([start.x, start.y, start.z, end.x, end.y, end.z]),
          3
        )
      )

      const material = new THREE.LineBasicMaterial({
        color: 0x94a3b8,
      })

      const line = new THREE.Line(geometry, material)
      line.userData.baseStart = start.clone()
      line.userData.baseEnd = end.clone()
      tetrahedron.add(line)
      edgesRef.current.push(line)
    })

    const light = new THREE.PointLight(0xffffff, 0.4)
    light.position.set(2, 2, 2)
    scene.add(light)

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.2)
    scene.add(ambientLight)

    let rotationX = 0
    let rotationY = 0

    const resize = () => {
      if (!containerRef.current || !cameraRef.current || !rendererRef.current) return
      const width = containerRef.current.clientWidth || window.innerWidth
      const height = containerRef.current.clientHeight || window.innerHeight
      cameraRef.current.aspect = width / Math.max(height, 1)
      cameraRef.current.updateProjectionMatrix()
      rendererRef.current.setSize(width, height, false)
    }

    const animate = () => {
      frameRef.current = requestAnimationFrame(animate)
      timeRef.current += 0.016

      const {
        data: liveData,
        phase: livePhase,
        escalationLevel: liveEscalationLevel,
        hasThresholdCrossed: liveThreshold,
        isApproachingFailure: liveApproachingFailure,
      } = propsRef.current

      const drift = liveData.interpolatedDrift
      const stability = liveData.interpolatedStability
      const coherence = liveData.interpolatedCoherence

      const targetCameraZ = liveEscalationLevel >= 3 ? 1.7 : 1.8
      const zoomTransition = 0.05
      if (cameraRef.current) {
        cameraRef.current.position.z +=
          (targetCameraZ - cameraRef.current.position.z) * zoomTransition
        cameraRef.current.updateProjectionMatrix()
      }

      const baseRotationSpeed = liveEscalationLevel >= 3 ? 0.004 : 0.002
      rotationX += baseRotationSpeed
      rotationY += baseRotationSpeed * 0.6

      const driftInfluence = drift * (0.2 + liveEscalationLevel * 0.05)
      rotationX += driftInfluence * 0.008
      rotationY += driftInfluence * 0.006

      const asymmetryAmount = liveApproachingFailure ? drift * 0.12 : 0
      const asymmetryX = Math.sin(timeRef.current * 0.7) * asymmetryAmount
      const asymmetryY = Math.cos(timeRef.current * 0.5) * asymmetryAmount * 0.6

      const breathingFreq = liveEscalationLevel >= 3 ? 1.2 : 0.5
      const depthBreathe = Math.sin(timeRef.current * breathingFreq) * 0.15
      const xBreathe = Math.cos(timeRef.current * 0.3) * 0.08
      tetrahedron.position.z = depthBreathe
      tetrahedron.position.x = xBreathe * drift + asymmetryX

      const snapIntensity = liveThreshold ? Math.sin(timeRef.current * 30) * 0.3 : 0
      const snapScale = 1 - snapIntensity * 0.08

      const breathingDamping = liveEscalationLevel >= 3 ? 0.02 : 0.08
      const baseBreathing = 1 + Math.sin(timeRef.current * breathingFreq) * breathingDamping
      const stabilityDamping = (0.05 + stability * 0.02) * snapScale
      const breathingScale = baseBreathing + stabilityDamping

      tetrahedron.rotation.x = rotationX + asymmetryY
      tetrahedron.rotation.y = rotationY
      tetrahedron.scale.set(breathingScale, breathingScale, breathingScale)

      const phaseColor = getPhaseColor(livePhase as any) || '#38BDF8'

      verticesRef.current.forEach((sphere, index) => {
        if (!(sphere.material instanceof THREE.MeshPhongMaterial)) return

        sphere.material.color.setStyle(phaseColor)
        sphere.material.emissive.setStyle(phaseColor)

        const phaseIntensity =
          {
            0: 0.25,
            1: 0.35,
            2: 0.5,
            3: 0.8,
          }[liveEscalationLevel] || 0.25

        const glowSpike = liveThreshold ? Math.sin(timeRef.current * 20) * 0.3 : 0
        const baseGlow = phaseIntensity + Math.sin(timeRef.current + index) * 0.1
        const flickerAmount = liveEscalationLevel >= 3 ? Math.random() * 0.1 : 0
        sphere.material.emissiveIntensity = baseGlow + glowSpike + flickerAmount

        const instabilityAmount = drift * (0.08 + liveEscalationLevel * 0.04)
        const wobbleFreq = liveEscalationLevel >= 3 ? 2.5 : 1.2
        const wobble = new THREE.Vector3(
          Math.sin(timeRef.current * wobbleFreq + index) * instabilityAmount,
          Math.cos(timeRef.current * wobbleFreq * 1.3 + index) * instabilityAmount,
          Math.sin(timeRef.current * 0.9 + index) * instabilityAmount * 0.5
        )

        const symmetryLoss = liveApproachingFailure
          ? new THREE.Vector3(
              (Math.random() - 0.5) * 0.04,
              (Math.random() - 0.5) * 0.04,
              (Math.random() - 0.5) * 0.02
            )
          : new THREE.Vector3()

        sphere.position.copy(sphere.userData.basePosition.clone().add(wobble).add(symmetryLoss))
      })

      edgesRef.current.forEach((line, index) => {
        if (!(line.material instanceof THREE.LineBasicMaterial)) return

        line.material.color.setStyle(phaseColor)

        const baseTension = drift * (0.08 + liveEscalationLevel * 0.03)
        const tensionFreq = liveEscalationLevel >= 3 ? 1.2 : 0.7
        const tensionWave =
          Math.sin(timeRef.current * tensionFreq + index * 0.5) *
          (0.04 + liveEscalationLevel * 0.02)

        const positions = line.geometry.attributes.position.array as Float32Array
        const baseStart: THREE.Vector3 = line.userData.baseStart
        const baseEnd: THREE.Vector3 = line.userData.baseEnd
        const direction = new THREE.Vector3().subVectors(baseEnd, baseStart).normalize()

        const stretchAmount = baseTension + tensionWave
        positions[0] = baseStart.x + direction.x * stretchAmount
        positions[1] = baseStart.y + direction.y * stretchAmount
        positions[2] = baseStart.z + direction.z * stretchAmount
        positions[3] = baseEnd.x - direction.x * stretchAmount
        positions[4] = baseEnd.y - direction.y * stretchAmount
        positions[5] = baseEnd.z - direction.z * stretchAmount
        line.geometry.attributes.position.needsUpdate = true
      })

      if (rendererRef.current && cameraRef.current && sceneRef.current) {
        rendererRef.current.render(sceneRef.current, cameraRef.current)
      }
    }

    resize()
    animate()

    const resizeObserver = new ResizeObserver(() => resize())
    resizeObserver.observe(container)
    window.addEventListener('resize', resize)

    return () => {
      window.removeEventListener('resize', resize)
      resizeObserver.disconnect()

      if (frameRef.current) {
        cancelAnimationFrame(frameRef.current)
      }

      scene.traverse((obj) => {
        const anyObj = obj as any
        if (anyObj.geometry) {
          anyObj.geometry.dispose()
        }
        if (anyObj.material) {
          if (Array.isArray(anyObj.material)) {
            anyObj.material.forEach((m: THREE.Material) => m.dispose())
          } else {
            anyObj.material.dispose()
          }
        }
      })

      renderer.dispose()
      renderer.forceContextLoss()

      if (renderer.domElement.parentNode) {
        renderer.domElement.parentNode.removeChild(renderer.domElement)
      }

      verticesRef.current = []
      edgesRef.current = []
      tetrahedronRef.current = null
      rendererRef.current = null
      cameraRef.current = null
      sceneRef.current = null
    }
  }, [])

  return (
    <div
      ref={containerRef}
      style={{
        width: '100%',
        height: '100%',
        position: 'relative',
      }}
    />
  )
}
