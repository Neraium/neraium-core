"use client";

import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { useMemo, useRef } from "react";
import * as THREE from "three";

function PositionPoint({ position }: { position: [number, number, number] }) {
  const ref = useRef<THREE.Mesh>(null);
  useFrame(() => {
    if (!ref.current) return;
    ref.current.position.lerp(new THREE.Vector3(...position), 0.18);
  });
  return (
    <mesh ref={ref}>
      <sphereGeometry args={[0.08, 20, 20]} />
      <meshStandardMaterial color="#f97316" emissive="#f97316" emissiveIntensity={0.3} />
    </mesh>
  );
}

export function TetrahedronPanel({ position }: { position: [number, number, number] }) {
  const vertices = useMemo(
    () =>
      [
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
      ] as [number, number, number][],
    []
  );

  return (
    <section className="panel tetra">
      <h3>System State</h3>
      <Canvas camera={{ position: [2.5, 2.3, 2.7], fov: 45 }}>
        <ambientLight intensity={0.6} />
        <pointLight position={[4, 3, 3]} intensity={1.2} />
        <lineSegments>
          <edgesGeometry args={[new THREE.TetrahedronGeometry(1.45)]} />
          <lineBasicMaterial color="#64748b" />
        </lineSegments>
        {vertices.map((v, i) => (
          <mesh key={i} position={v}>
            <sphereGeometry args={[0.05, 16, 16]} />
            <meshStandardMaterial color="#93c5fd" />
          </mesh>
        ))}
        <PositionPoint position={position} />
        <OrbitControls enablePan={false} minDistance={2} maxDistance={5} />
      </Canvas>
    </section>
  );
}
