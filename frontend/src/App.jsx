/**
 * Spectra Vis - Main App Component
 * Copyright (c) 2025 3rdEyeVisuals
 */

import React, { useState, useEffect } from 'react'
import TensorGrid3D from './components/TensorGrid3D'
import ControlPanel from './components/ControlPanel'
import StatusBar from './components/StatusBar'

const styles = {
  container: {
    width: '100vw',
    height: '100vh',
    display: 'flex',
    flexDirection: 'column',
    background: 'linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 100%)',
    color: '#fff',
  },
  header: {
    padding: '12px 20px',
    background: 'rgba(0,0,0,0.4)',
    borderBottom: '1px solid rgba(255,255,255,0.1)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  title: {
    fontSize: '20px',
    fontWeight: 600,
    background: 'linear-gradient(90deg, #50c878, #4a90d9)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
  },
  author: {
    fontSize: '12px',
    color: 'rgba(255,255,255,0.5)',
  },
  main: {
    flex: 1,
    display: 'flex',
    overflow: 'hidden',
  },
  sidebar: {
    width: '300px',
    background: 'rgba(0,0,0,0.3)',
    borderRight: '1px solid rgba(255,255,255,0.1)',
    padding: '16px',
    overflowY: 'auto',
  },
  canvas: {
    flex: 1,
    position: 'relative',
  },
}

function App() {
  const [serverStatus, setServerStatus] = useState('checking')
  const [models, setModels] = useState([])
  const [files, setFiles] = useState([])
  const [selectedModel, setSelectedModel] = useState('llama')
  const [selectedSize, setSelectedSize] = useState('7b')
  const [gridData, setGridData] = useState(null)
  const [loadedFile, setLoadedFile] = useState(null)
  const [error, setError] = useState(null)

  // Check server status on mount
  useEffect(() => {
    checkServer()
  }, [])

  const checkServer = async () => {
    try {
      const res = await fetch('/api/status')
      if (res.ok) {
        const data = await res.json()
        setServerStatus('connected')
        if (data.loaded) {
          setLoadedFile(data.filepath)
        }
        // Fetch models and files
        fetchModels()
        fetchFiles()
      } else {
        setServerStatus('error')
      }
    } catch (e) {
      setServerStatus('disconnected')
      setError('Cannot connect to backend server. Make sure it is running on port 8000.')
    }
  }

  const fetchModels = async () => {
    try {
      const res = await fetch('/api/models')
      const data = await res.json()
      setModels(data.profiles || {})
    } catch (e) {
      console.error('Failed to fetch models:', e)
    }
  }

  const fetchFiles = async () => {
    try {
      const res = await fetch('/api/files')
      const data = await res.json()
      setFiles(data.files || [])
    } catch (e) {
      console.error('Failed to fetch files:', e)
    }
  }

  const loadFile = async (filepath) => {
    try {
      setError(null)
      const res = await fetch('/api/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filepath }),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Failed to load file')
      }
      const data = await res.json()
      setLoadedFile(filepath)
      // Auto-analyze after loading
      analyzeData()
    } catch (e) {
      setError(e.message)
    }
  }

  const analyzeData = async () => {
    try {
      setError(null)
      const res = await fetch('/api/tensor-grid', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_family: selectedModel,
          model_size: selectedSize,
        }),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Failed to analyze data')
      }
      const data = await res.json()
      setGridData(data)
    } catch (e) {
      setError(e.message)
    }
  }

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <div>
          <div style={styles.title}>Spectra Vis</div>
          <div style={styles.author}>Tensor Visualization by 3rdEyeVisuals</div>
        </div>
        <StatusBar status={serverStatus} loadedFile={loadedFile} />
      </header>

      <main style={styles.main}>
        <aside style={styles.sidebar}>
          <ControlPanel
            models={models}
            files={files}
            selectedModel={selectedModel}
            selectedSize={selectedSize}
            onModelChange={setSelectedModel}
            onSizeChange={setSelectedSize}
            onLoadFile={loadFile}
            onAnalyze={analyzeData}
            onRefreshFiles={fetchFiles}
            error={error}
            gridData={gridData}
          />
        </aside>

        <div style={styles.canvas}>
          <TensorGrid3D data={gridData} />
        </div>
      </main>
    </div>
  )
}

export default App
