/**
 * Spectra Vis - 3D Tensor Grid Visualization
 * Copyright (c) 2025 3rdEyeVisuals
 */

import { useRef, useMemo, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, Text } from '@react-three/drei'

// Individual tensor block - using instanced rendering for performance
function TensorBlock({ position, color, intensity, label, onHover, scale = 1 }) {
  const meshRef = useRef()
  const baseScale = 0.06 * scale
  const heightScale = Math.max(0.02, intensity * 0.15) * scale

  return (
    <group position={position}>
      <mesh
        ref={meshRef}
        onPointerOver={(e) => {
          e.stopPropagation()
          onHover && onHover(label)
        }}
        onPointerOut={() => onHover && onHover(null)}
      >
        <boxGeometry args={[baseScale, heightScale, baseScale]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.3 + intensity * 0.4}
          metalness={0.2}
          roughness={0.6}
        />
      </mesh>
    </group>
  )
}

// Layer label
function LayerLabel({ position, text }) {
  return (
    <Text
      position={position}
      fontSize={0.25}
      color="#ffffff"
      anchorX="right"
      anchorY="middle"
    >
      {text}
    </Text>
  )
}

// Tensor type label (column header) - kept for future use
// function TypeLabel({ position, text, color }) {
//   return (
//     <Text
//       position={position}
//       fontSize={0.2}
//       color={color || "#aaaaaa"}
//       anchorX="center"
//       anchorY="bottom"
//       rotation={[-Math.PI / 6, 0, 0]}
//     >
//       {text}
//     </Text>
//   )
// }

// Main grid scene
function TensorScene({ data, autoRotate }) {
  const groupRef = useRef()
  const [hoveredTensor, setHoveredTensor] = useState(null)

  // Group tensors by layer - MUST be before any conditional returns
  const layerGroups = useMemo(() => {
    if (!data || !data.grid || data.grid.length === 0) return {}
    const groups = {}
    data.grid.forEach((t, idx) => {
      const layer = t.layer
      if (!groups[layer]) groups[layer] = []
      groups[layer].push({ ...t, idx })
    })
    return groups
  }, [data])

  // Auto-rotate slowly (controlled by prop)
  useFrame(() => {
    if (groupRef.current && autoRotate) {
      groupRef.current.rotation.y += 0.002
    }
  })

  // Early return AFTER all hooks
  if (!data || !data.grid || data.grid.length === 0) {
    return (
      <Text
        position={[0, 0, 0]}
        fontSize={0.5}
        color="#666666"
        anchorX="center"
        anchorY="middle"
      >
        Load data to visualize
      </Text>
    )
  }

  // Calculate grid dimensions based on ACTUAL data
  const layers = Object.keys(layerGroups).map(Number).sort((a, b) => {
    // Special sorting: -1 first, then 0-N, then 998, 999
    if (a < 0) return -1
    if (b < 0) return 1
    if (a >= 900 && b < 900) return 1
    if (b >= 900 && a < 900) return -1
    return a - b
  })

  // Find max tensors per layer to determine grid width
  const maxTensorsPerLayer = Math.max(...layers.map(l => (layerGroups[l] || []).length), 1)

  // Dynamic spacing based on data size - ensure no overlap
  const spacing = 0.1  // Fixed small spacing between tensors
  const layerSpacing = 0.18  // Tighter spacing between layers

  const gridWidth = maxTensorsPerLayer * spacing
  const gridHeight = layers.length * layerSpacing

  // Scale factor for blocks based on density
  const blockScale = Math.min(1, 30 / maxTensorsPerLayer)

  return (
    <group ref={groupRef} position={[0, 0, 0]}>
      {/* Lighting */}
      <ambientLight intensity={0.5} />
      <pointLight position={[30, 30, 30]} intensity={0.8} />
      <pointLight position={[-30, -30, -30]} intensity={0.4} color="#4a90d9" />
      <directionalLight position={[0, 20, 10]} intensity={0.3} />

      {/* Render tensor blocks */}
      {layers.map((layer, layerIdx) => {
        const tensors = layerGroups[layer] || []
        const y = (layerIdx - layers.length / 2) * layerSpacing

        return (
          <group key={layer}>
            {/* Layer label */}
            <LayerLabel
              position={[-gridWidth / 2 - 1.2, y, 0]}
              text={layer === -1 ? 'Emb' : layer >= 998 ? (layer === 998 ? 'Norm' : 'Out') : `L${layer}`}
            />

            {/* Tensor blocks for this layer */}
            {tensors.map((tensor, idx) => {
              const x = (idx - tensors.length / 2 + 0.5) * spacing
              return (
                <TensorBlock
                  key={`${layer}-${idx}`}
                  position={[x, y, 0]}
                  color={tensor.color || '#50c878'}
                  intensity={tensor.intensity || 0.5}
                  label={`Layer ${layer}: ${tensor.type || `tensor_${idx}`} (${(tensor.intensity * 100).toFixed(0)}%)`}
                  onHover={setHoveredTensor}
                  scale={blockScale}
                />
              )
            })}
          </group>
        )
      })}

      {/* Hover info display */}
      {hoveredTensor && (
        <Text
          position={[0, -gridHeight / 2 - 2, 0]}
          fontSize={0.4}
          color="#ffffff"
          anchorX="center"
          anchorY="middle"
          outlineWidth={0.03}
          outlineColor="#000000"
        >
          {hoveredTensor}
        </Text>
      )}

      {/* Grid floor for reference */}
      <gridHelper
        args={[Math.max(gridWidth, gridHeight) + 4, 20, '#333333', '#222222']}
        position={[0, -gridHeight / 2 - 0.8, 0]}
        rotation={[0, 0, 0]}
      />
    </group>
  )
}

// Main component with controls
function TensorGrid3D({ data }) {
  const [autoRotate, setAutoRotate] = useState(false)

  // Calculate camera distance based on data size
  const cameraDistance = useMemo(() => {
    if (!data || !data.grid) return 30
    const numLayers = data.total_layers || 32
    const numTensors = data.grid.length || 100
    const tensorsPerLayer = numTensors / numLayers
    // Camera needs to be far enough to see the whole grid
    return Math.max(30, Math.max(numLayers * 0.6, tensorsPerLayer * 0.4))
  }, [data])

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      {/* Rotation toggle button */}
      <button
        onClick={() => setAutoRotate(!autoRotate)}
        style={{
          position: 'absolute',
          top: '10px',
          right: '10px',
          zIndex: 100,
          padding: '8px 16px',
          background: autoRotate ? '#50c878' : '#333333',
          color: '#ffffff',
          border: 'none',
          borderRadius: '4px',
          cursor: 'pointer',
          fontSize: '12px',
          fontWeight: 'bold',
        }}
      >
        {autoRotate ? 'Stop Rotation' : 'Auto Rotate'}
      </button>

      <Canvas
        camera={{ position: [0, 0, cameraDistance], fov: 50 }}
        style={{ background: '#0a0a0f' }}
      >
        <TensorScene data={data} autoRotate={autoRotate} />
        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          minDistance={5}
          maxDistance={100}
          zoomSpeed={1.2}
          panSpeed={0.8}
          rotateSpeed={0.5}
        />
      </Canvas>
    </div>
  )
}

export default TensorGrid3D
