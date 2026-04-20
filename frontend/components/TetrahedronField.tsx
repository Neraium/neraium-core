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
  const timeRef = useRef(0)

  useEffect(() => {
    if (!containerRef.current) return

    // Scene setup
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(0x0a0e1a)
    sceneRef.current = scene

    const camera = new THREE.PerspectiveCamera(
      75,
      window.innerWidth / window.innerHeight,
      0.1,
      1000
    )
    // Camera positioned closer for larger tetrahedron fill
    // Will zoom in (z decrease) during critical phase
    cameraRef.current = camera

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    renderer.setSize(window.innerWidth, window.innerHeight)
    renderer.setPixelRatio(window.devicePixelRatio)
    containerRef.current.appendChild(renderer.domElement)
    rendererRef.current = renderer

    // Create tetrahedron group
    const tetrahedron = new THREE.Group()
    tetrahedronRef.current = tetrahedron
    scene.add(tetrahedron)

    // Base tetrahedron vertices (scaled larger for 80-85% fill)
    const baseScale = 1.35
    const vertices = [
      new THREE.Vector3(0, 1, 0), // top
      new THREE.Vector3(-0.866, -0.5, 0), // bottom-left
      new THREE.Vector3(0.866, -0.5, 0), // bottom-right
      new THREE.Vector3(0, -0.2, 0.8), // back
    ]

    // Create vertex spheres
    const sphereGeometry = new THREE.SphereGeometry(0.12, 16, 16)
    vertices.forEach((vertex, index) => {
      const material = new THREE.MeshPhongMaterial({
        color: 0x38BDF8,
        emissive: 0x38BDF8,
        emissiveIntensity: 0.3,
      })
      const sphere = new THREE.Mesh(sphereGeometry, material)
      sphere.position.copy(vertex.multiplyScalar(baseScale))
      sphere.userData.basePosition = sphere.position.clone()
      tetrahedron.add(sphere)
      verticesRef.current.push(sphere)
    })

    // Create edges with deformation data
    const edges = [
      [0, 1],
      [1, 2],
      [2, 0],
      [0, 3],
      [1, 3],
      [2, 3],
    ]

    edges.forEach(([startIdx, endIdx]) => {
      const geometry = new THREE.BufferGeometry()
      const start = vertices[startIdx].clone().multiplyScalar(baseScale)
      const end = vertices[endIdx].clone().multiplyScalar(baseScale)
      geometry.setAttribute('position', new THREE.BufferAttribute(
        new Float32Array([
          start.x, start.y, start.z,
          end.x, end.y, end.z,
        ]),
        3
      ))

      const material = new THREE.LineBasicMaterial({
        color: 0x94a3b8,
        linewidth: 2,
      })
      const line = new THREE.Line(geometry, material)
      line.userData.baseStart = start
      line.userData.baseEnd = end
      tetrahedron.add(line)
      edgesRef.current.push(line)
    })

    // Light setup
    const light = new THREE.PointLight(0xffffff, 0.4)
    light.position.set(2, 2, 2)
    scene.add(light)

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.2)
    scene.add(ambientLight)

    // Animation loop
    let animationFrameId: number
    let rotationX = 0
    let rotationY = 0

    const animate = () => {
      animationFrameId = requestAnimationFrame(animate)
      timeRef.current += 0.016 // ~60fps

      // CAMERA ZOOM FOR CRITICAL STATES (2-5% closer)
      const targetCameraZ = escalationLevel >= 3 ? 1.7 : 1.8 // Zoom in at critical
      const zoomTransition = 0.05 // Smooth camera movement
      if (cameraRef.current) {
        cameraRef.current.position.z +=
          (targetCameraZ - cameraRef.current.position.z) * zoomTransition
        cameraRef.current.updateProjectionMatrix()
      }

      // CONTINUOUS MOTION (never idle)
      const baseRotationSpeed = escalationLevel >= 3 ? 0.004 : 0.002 // Faster at critical
      rotationX += baseRotationSpeed
      rotationY += baseRotationSpeed * 0.6

      // Drift-influenced rotation (adds instability)
      const driftInfluence = data.interpolatedDrift * (0.2 + escalationLevel * 0.05)
      rotationX += driftInfluence * 0.008
      rotationY += driftInfluence * 0.006

      // CONSEQUENCE FEELING: Asymmetry approaching failure
      const asymmetryAmount = isApproachingFailure ? data.interpolatedDrift * 0.12 : 0
      const asymmetryX = Math.sin(timeRef.current * 0.7) * asymmetryAmount
      const asymmetryY = Math.cos(timeRef.current * 0.5) * asymmetryAmount * 0.6

      // Depth breathing (Z-axis parallax effect)
      // Reduced smoothness at critical (choppier motion)
      const breathingFreq = escalationLevel >= 3 ? 1.2 : 0.5
      const depthBreathe = Math.sin(timeRef.current * breathingFreq) * 0.15
      const xBreathe = Math.cos(timeRef.current * 0.3) * 0.08
      tetrahedron.position.z = depthBreathe
      tetrahedron.position.x = xBreathe * data.interpolatedDrift + asymmetryX

      // SNAP EFFECT on threshold cross (brief tightening)
      const snapIntensity = hasThresholdCrossed ? Math.sin(timeRef.current * 30) * 0.3 : 0
      const snapScale = 1 - snapIntensity * 0.08 // Tighten briefly

      // Scale: subtle breathing, never perfectly still
      // At critical: less smooth, more tense
      const breathingDamping = escalationLevel >= 3 ? 0.02 : 0.08
      const baseBreathing = 1 + Math.sin(timeRef.current * breathingFreq) * breathingDamping
      const stabilityDamping = (0.05 + data.interpolatedStability * 0.02) * snapScale
      const breathingScale = baseBreathing + stabilityDamping

      tetrahedron.rotation.x = rotationX + asymmetryY
      tetrahedron.rotation.y = rotationY
      tetrahedron.scale.set(breathingScale, breathingScale, breathingScale)

      // Color based on phase
      const phaseColor = getPhaseColor(phase as any) || '#38BDF8'

      // Update vertices with decision gravity effects
      verticesRef.current.forEach((sphere, index) => {
        if (sphere.material instanceof THREE.MeshPhongMaterial) {
          sphere.material.color.setStyle(phaseColor)
          sphere.material.emissive.setStyle(phaseColor)

          // INTENSITY INCREASES AT CRITICAL
          const phaseIntensity = {
            0: 0.25, // Stable
            1: 0.35, // Drift forming
            2: 0.5, // Instability forming
            3: 0.8, // Critical (increased)
          }[escalationLevel] || 0.25

          // GLOW SPIKE on threshold cross
          const glowSpike = hasThresholdCrossed ? Math.sin(timeRef.current * 20) * 0.3 : 0
          const baseGlow = phaseIntensity + Math.sin(timeRef.current + index) * 0.1

          // Optional micro-flicker in extreme instability
          const flickerAmount = escalationLevel >= 3 ? Math.random() * 0.1 : 0
          sphere.material.emissiveIntensity = baseGlow + glowSpike + flickerAmount

          // MICRO-INSTABILITY with increased chaos at critical
          const instabilityAmount =
            data.interpolatedDrift * (0.08 + escalationLevel * 0.04)
          const wobbleFreq = escalationLevel >= 3 ? 2.5 : 1.2
          const wobble = new THREE.Vector3(
            Math.sin(timeRef.current * wobbleFreq + index) * instabilityAmount,
            Math.cos(timeRef.current * wobbleFreq * 1.3 + index) * instabilityAmount,
            Math.sin(timeRef.current * 0.9 + index) * instabilityAmount * 0.5
          )

          // LOSS OF SYMMETRY approaching failure
          const symmetryLoss = isApproachingFailure
            ? new THREE.Vector3(
                (Math.random() - 0.5) * 0.04,
                (Math.random() - 0.5) * 0.04,
                (Math.random() - 0.5) * 0.02
              )
            : new THREE.Vector3()

          sphere.position.copy(
            sphere.userData.basePosition
              .clone()
              .add(wobble)
              .add(symmetryLoss)
          )
        }
      })

      // Update edges with tension behavior (increased at critical)
      edgesRef.current.forEach((line, index) => {
        if (line.material instanceof THREE.LineBasicMaterial) {
          line.material.color.setStyle(phaseColor)

          // EDGE TENSION increases at critical (spikes to show stress)
          const baseTension = data.interpolatedDrift * (0.08 + escalationLevel * 0.03)
          const tensionFreq = escalationLevel >= 3 ? 1.2 : 0.7
          const tensionWave = Math.sin(timeRef.current * tensionFreq + index * 0.5) * (0.04 + escalationLevel * 0.02)

          // Deform edge by stretching
          const positions = line.geometry.attributes.position.array as Float32Array
          const baseStart = line.userData.baseStart
          const baseEnd = line.userData.baseEnd
          const direction = new THREE.Vector3().subVectors(baseEnd, baseStart).normalize()

          const stretchAmount = baseTension + tensionWave
          positions[0] = baseStart.x + direction.x * stretchAmount
          positions[1] = baseStart.y + direction.y * stretchAmount
          positions[2] = baseStart.z + direction.z * stretchAmount
          positions[3] = baseEnd.x - direction.x * stretchAmount
          positions[4] = baseEnd.y - direction.y * stretchAmount
          positions[5] = baseEnd.z - direction.z * stretchAmount
          line.geometry.attributes.position.needsUpdate = true
        }
      })

      renderer.render(scene, camera)
    }

    animate()

    // Handle window resize
    const handleResize = () => {
      const width = window.innerWidth
      const height = window.innerHeight

      camera.aspect = width / height
      camera.updateProjectionMatrix()

      renderer.setSize(width, height)
    }

    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      cancelAnimationFrame(animationFrameId)
      renderer.dispose()
      renderer.domElement.remove()
    }
  }, [data, phaseProgress, phase])

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
