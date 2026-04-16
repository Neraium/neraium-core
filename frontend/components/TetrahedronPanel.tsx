"use client";

import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useMemo, useRef, useEffect } from "react";
import * as THREE from "three";

type PositionPointProps = {
  position: [number, number, number];
  confidence: number;
};

function PositionPoint({ position, confidence }: PositionPointProps) {
  const ref = useRef<THREE.Mesh>(null);
  const groupRef = useRef<THREE.Group>(null);

  useFrame(() => {
    if (!ref.current) return;
    // Smooth position interpolation
    ref.current.position.lerp(new THREE.Vector3(...position), 0.12);

    // Pulse animation based on confidence
    if (groupRef.current) {
      const time = Date.now() * 0.001;
      const confidenceFactor = Math.max(0.6, confidence); // Min 60% size
      const basePulse = 0.08 * confidenceFactor;
      const pulse = basePulse + Math.sin(time * 2.5) * basePulse * 0.4;
      groupRef.current.children[0].scale.set(pulse, pulse, pulse);

      // Glow intensity based on confidence
      const glowMaterial = groupRef.current.children[0].material as THREE.MeshStandardMaterial;
      if (glowMaterial) {
        glowMaterial.emissiveIntensity = 0.3 + confidence * 0.7;
      }
    }
  });

  return (
    <group ref={groupRef}>
      <mesh ref={ref}>
        <sphereGeometry args={[0.08, 24, 24]} />
        <meshStandardMaterial
          color="#f97316"
          emissive="#f97316"
          emissiveIntensity={0.5 + confidence * 0.5}
          metalness={0.3}
          roughness={0.4}
        />
      </mesh>
    </group>
  );
}

type VertexProps = {
  position: [number, number, number];
  currentPos: [number, number, number];
  confidence: number;
};

function TetrahedronVertex({ position, currentPos, confidence }: VertexProps) {
  const meshRef = useRef<THREE.Mesh>(null);

  useEffect(() => {
    // Calculate distance from current position
    const dx = position[0] - currentPos[0];
    const dy = position[1] - currentPos[1];
    const dz = position[2] - currentPos[2];
    const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

    // Distance-based opacity: closer = brighter
    const maxDistance = Math.sqrt(2 * 2 * 2) * 2; // Max possible distance in our space
    const proximity = Math.max(0, 1 - distance / maxDistance);
    const baseOpacity = 0.3 + proximity * 0.7; // 0.3 - 1.0
    const opacity = baseOpacity * (0.7 + confidence * 0.3); // Modulate by confidence

    if (meshRef.current) {
      const material = meshRef.current.material as THREE.MeshStandardMaterial;
      material.opacity = opacity;
      material.emissiveIntensity = proximity * 0.5; // Glow closer vertices
    }
  }, [currentPos, confidence, position]);

  return (
    <mesh ref={meshRef} position={position}>
      <sphereGeometry args={[0.06, 16, 16]} />
      <meshStandardMaterial
        color="#93c5fd"
        transparent
        opacity={0.8}
        metalness={0.5}
        roughness={0.3}
        emissive="#60a5fa"
        emissiveIntensity={0.2}
      />
    </mesh>
  );
}

type EdgeProps = {
  start: [number, number, number];
  end: [number, number, number];
  currentPos: [number, number, number];
  confidence: number;
};

function TetrahedronEdge({ start, end, currentPos, confidence }: EdgeProps) {
  const lineRef = useRef<THREE.LineSegments>(null);

  useEffect(() => {
    if (!lineRef.current) return;

    // Calculate proximity of edge to current position
    const midX = (start[0] + end[0]) / 2;
    const midY = (start[1] + end[1]) / 2;
    const midZ = (start[2] + end[2]) / 2;

    const dx = midX - currentPos[0];
    const dy = midY - currentPos[1];
    const dz = midZ - currentPos[2];
    const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

    const maxDistance = Math.sqrt(2 * 2 * 2) * 2;
    const proximity = Math.max(0, 1 - distance / maxDistance);
    const opacity = (0.2 + proximity * 0.6) * (0.7 + confidence * 0.3);

    if (lineRef.current.material instanceof THREE.LineBasicMaterial) {
      lineRef.current.material.opacity = opacity;
    }
  }, [currentPos, confidence, start, end]);

  const points = [new THREE.Vector3(...start), new THREE.Vector3(...end)];
  const geometry = new THREE.BufferGeometry().setFromPoints(points);

  return (
    <lineSegments ref={lineRef} geometry={geometry}>
      <lineBasicMaterial
        color="#64748b"
        transparent
        opacity={0.5}
        linewidth={1}
      />
    </lineSegments>
  );
}

type Props = {
  position: [number, number, number];
  confidence?: number;
};

export function TetrahedronPanel({ position, confidence = 0.5 }: Props) {
  const vertices: [number, number, number][] = useMemo(
    () => [
      [1, 1, 1],
      [1, -1, -1],
      [-1, 1, -1],
      [-1, -1, 1],
    ],
    []
  );

  const edges = useMemo(
    () => [
      [vertices[0], vertices[1]],
      [vertices[0], vertices[2]],
      [vertices[0], vertices[3]],
      [vertices[1], vertices[2]],
      [vertices[1], vertices[3]],
      [vertices[2], vertices[3]],
    ],
    [vertices]
  );

  return (
    <section className="panel tetra">
      <h3>System State · Tetrahedral Geometry</h3>
      <Canvas camera={{ position: [2.5, 2.3, 2.7], fov: 45 }}>
        {/* Enhanced lighting setup */}
        <ambientLight intensity={0.5} />
        <pointLight position={[4, 3, 3]} intensity={1.2} color="#ffffff" />
        <pointLight position={[-3, -2, -3]} intensity={0.5} color="#60a5fa" decay={2} />

        {/* Tetrahedron structure */}
        <group>
          {/* Edges with dynamic opacity */}
          {edges.map((edge, idx) => (
            <TetrahedronEdge
              key={`edge-${idx}`}
              start={edge[0] as [number, number, number]}
              end={edge[1] as [number, number, number]}
              currentPos={position}
              confidence={confidence}
            />
          ))}

          {/* Vertices with distance-based emphasis */}
          {vertices.map((v, i) => (
            <TetrahedronVertex
              key={`vertex-${i}`}
              position={v}
              currentPos={position}
              confidence={confidence}
            />
          ))}
        </group>

        {/* Current position with confidence-based intensity */}
        <PositionPoint position={position} confidence={confidence} />

        {/* Orbit controls */}
        <OrbitControls
          enablePan={false}
          minDistance={2}
          maxDistance={5}
          autoRotate={false}
          dampingFactor={0.05}
        />
      </Canvas>

      {/* Status indicator */}
      <div className="tetra-status">
        <div className="confidence-indicator">
          <span className="confidence-label">Confidence</span>
          <div className="confidence-bar">
            <div
              className="confidence-fill"
              style={{ width: `${confidence * 100}%` }}
            />
          </div>
          <span className="confidence-value">{Math.round(confidence * 100)}%</span>
        </div>
      </div>
    </section>
  );
}
