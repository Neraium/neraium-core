'use client'

import React, { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { getPhaseColor, getPhaseIntensity } from '@/lib/phaseController'

interface TetrahedronData {
  interpolatedDrift: number
  interpolatedStability: number
  interpolatedCoherence: number
  interpolatedConfidence: number
}

interface TetrahedronFieldProps {
  data: TetrahedronData
  phaseProgress: number
}

export function TetrahedronField({ data, phaseProgress }: TetrahedronFieldProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<THREE.Scene | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const tetrahedronRef = useRef<THREE.Group | null>(null)
  const edgesRef = useRef<THREE.Line[]>([])
  const verticesRef = useRef<THREE.Mesh[]>([])

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
    camera.position.z = 2.5
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

    // Tetrahedron vertices (normalized)
    const vertices = [
      new THREE.Vector3(0, 1, 0), // top
      new THREE.Vector3(-0.866, -0.5, 0), // bottom-left
      new THREE.Vector3(0.866, -0.5, 0), // bottom-right
      new THREE.Vector3(0, -0.2, 0.8), // back
    ]

    // Create vertex spheres
    const sphereGeometry = new THREE.SphereGeometry(0.08, 16, 16)
    vertices.forEach((vertex, index) => {
      const material = new THREE.MeshPhongMaterial({
        color: 0x38BDF8,
        emissive: 0x38BDF8,
        emissiveIntensity: 0.3,
      })
      const sphere = new THREE.Mesh(sphereGeometry, material)
      sphere.position.copy(vertex)
      tetrahedron.add(sphere)
      verticesRef.current.push(sphere)
    })

    // Create edges
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
      geometry.setAttribute('position', new THREE.BufferAttribute(
        new Float32Array([
          vertices[startIdx].x, vertices[startIdx].y, vertices[startIdx].z,
          vertices[endIdx].x, vertices[endIdx].y, vertices[endIdx].z,
        ]),
        3
      ))

      const material = new THREE.LineBasicMaterial({
        color: 0x94a3b8,
        linewidth: 2,
      })
      const line = new THREE.Line(geometry, material)
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

      // Smooth rotation based on drift
      const driftInfluence = data.interpolatedDrift * 0.3
      rotationX += 0.005 + driftInfluence * 0.01
      rotationY += 0.003 + driftInfluence * 0.008

      // Add breathing effect at stable phase
      const breathingScale = 1 + Math.sin(phaseProgress * Math.PI * 8) * 0.05

      tetrahedron.rotation.x = rotationX
      tetrahedron.rotation.y = rotationY
      tetrahedron.scale.set(breathingScale, breathingScale, breathingScale)

      // Update vertex colors based on drift
      const driftColor = new THREE.Color()
      const baseColor = getPhaseColor('Stable')
      driftColor.setStyle(baseColor)

      verticesRef.current.forEach((sphere) => {
        if (sphere.material instanceof THREE.MeshPhongMaterial) {
          sphere.material.color.setStyle(baseColor)
          sphere.material.emissive.setStyle(baseColor)

          // Intensity glow effect
          const intensity = getPhaseIntensity('Stable', phaseProgress)
          sphere.material.emissiveIntensity = 0.3 + intensity * 0.4
        }
      })

      // Update edge colors with drift tension
      edgesRef.current.forEach((line, index) => {
        if (line.material instanceof THREE.LineBasicMaterial) {
          const tension = data.interpolatedDrift * 0.5
          line.material.color.setStyle(baseColor)
          line.material.linewidth = 1.5 + tension
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
  }, [data, phaseProgress])

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
