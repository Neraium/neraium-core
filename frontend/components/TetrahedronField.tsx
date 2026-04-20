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
}

export function TetrahedronField({ data, phaseProgress, phase }: TetrahedronFieldProps) {
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
    camera.position.z = 1.8
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

      // Continuous motion (never idle) - base rotation always increases
      const baseRotationSpeed = 0.002
      rotationX += baseRotationSpeed
      rotationY += baseRotationSpeed * 0.6

      // Drift-influenced rotation (adds instability)
      const driftInfluence = data.interpolatedDrift * 0.2
      rotationX += driftInfluence * 0.008
      rotationY += driftInfluence * 0.006

      // Depth breathing (Z-axis parallax effect)
      const depthBreathe = Math.sin(timeRef.current * 0.5) * 0.15
      const xBreathe = Math.cos(timeRef.current * 0.3) * 0.08
      tetrahedron.position.z = depthBreathe
      tetrahedron.position.x = xBreathe * data.interpolatedDrift

      // Scale: subtle breathing, never perfectly still
      const baseBreathing = 1 + Math.sin(timeRef.current * 0.8) * 0.03
      const stabilityDamping = 0.05 + data.interpolatedStability * 0.02
      const breathingScale = baseBreathing + stabilityDamping

      tetrahedron.rotation.x = rotationX
      tetrahedron.rotation.y = rotationY
      tetrahedron.scale.set(breathingScale, breathingScale, breathingScale)

      // Color based on phase
      const phaseColor = getPhaseColor(phase as any) || '#38BDF8'

      // Update vertices with micro-instability (subtle asymmetry under drift)
      verticesRef.current.forEach((sphere, index) => {
        if (sphere.material instanceof THREE.MeshPhongMaterial) {
          sphere.material.color.setStyle(phaseColor)
          sphere.material.emissive.setStyle(phaseColor)

          // Phase-based intensity
          const phaseIntensity = {
            'Stable': 0.25,
            'Drift forming': 0.35,
            'Instability forming': 0.5,
            'Critical': 0.7,
          }[phase] || 0.25

          sphere.material.emissiveIntensity = phaseIntensity + Math.sin(timeRef.current + index) * 0.1

          // Micro-instability: slight wobble on vertices under drift
          const instabilityAmount = data.interpolatedDrift * 0.08
          const wobble = new THREE.Vector3(
            Math.sin(timeRef.current * 1.2 + index) * instabilityAmount,
            Math.cos(timeRef.current * 1.5 + index) * instabilityAmount,
            Math.sin(timeRef.current * 0.9 + index) * instabilityAmount * 0.5
          )
          sphere.position.copy(sphere.userData.basePosition.clone().add(wobble))
        }
      })

      // Update edges with tension behavior
      edgesRef.current.forEach((line, index) => {
        if (line.material instanceof THREE.LineBasicMaterial) {
          line.material.color.setStyle(phaseColor)

          // Edge tension: stretch/compress based on drift
          const tension = data.interpolatedDrift * 0.08
          const tensionWave = Math.sin(timeRef.current * 0.7 + index * 0.5) * 0.04

          // Deform edge by stretching
          const positions = line.geometry.attributes.position.array as Float32Array
          const baseStart = line.userData.baseStart
          const baseEnd = line.userData.baseEnd
          const direction = new THREE.Vector3().subVectors(baseEnd, baseStart).normalize()

          const stretchAmount = tension + tensionWave
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
