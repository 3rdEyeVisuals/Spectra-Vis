/**
 * Spectra Vis - Control Panel Component
 * Copyright (c) 2025 3rdEyeVisuals
 */

import React from 'react'

const styles = {
  section: {
    marginBottom: '24px',
  },
  sectionTitle: {
    fontSize: '12px',
    fontWeight: 600,
    color: 'rgba(255,255,255,0.6)',
    textTransform: 'uppercase',
    letterSpacing: '1px',
    marginBottom: '12px',
  },
  select: {
    width: '100%',
    padding: '10px 12px',
    background: '#1a1a2e',
    border: '1px solid rgba(255,255,255,0.2)',
    borderRadius: '6px',
    color: '#fff',
    fontSize: '14px',
    marginBottom: '8px',
    cursor: 'pointer',
    outline: 'none',
  },
  option: {
    background: '#1a1a2e',
    color: '#fff',
    padding: '8px',
  },
  button: {
    width: '100%',
    padding: '12px',
    background: 'linear-gradient(135deg, #50c878, #3da861)',
    border: 'none',
    borderRadius: '6px',
    color: '#fff',
    fontSize: '14px',
    fontWeight: 600,
    cursor: 'pointer',
    marginBottom: '8px',
    transition: 'all 0.2s',
  },
  buttonUpload: {
    width: '100%',
    padding: '12px',
    background: 'linear-gradient(135deg, #4a90d9, #357abd)',
    border: 'none',
    borderRadius: '6px',
    color: '#fff',
    fontSize: '14px',
    fontWeight: 600,
    cursor: 'pointer',
    marginBottom: '8px',
    transition: 'all 0.2s',
  },
  buttonSecondary: {
    width: '100%',
    padding: '10px',
    background: 'rgba(255,255,255,0.1)',
    border: '1px solid rgba(255,255,255,0.2)',
    borderRadius: '6px',
    color: '#fff',
    fontSize: '13px',
    cursor: 'pointer',
    marginBottom: '8px',
    transition: 'all 0.2s',
  },
  fileList: {
    maxHeight: '200px',
    overflowY: 'auto',
    marginBottom: '12px',
  },
  fileItem: {
    padding: '8px 12px',
    background: 'rgba(255,255,255,0.05)',
    borderRadius: '4px',
    marginBottom: '4px',
    cursor: 'pointer',
    fontSize: '13px',
    transition: 'all 0.2s',
    wordBreak: 'break-all',
  },
  fileItemHover: {
    background: 'rgba(80,200,120,0.2)',
  },
  error: {
    padding: '12px',
    background: 'rgba(255,100,100,0.2)',
    border: '1px solid rgba(255,100,100,0.3)',
    borderRadius: '6px',
    color: '#ff6b6b',
    fontSize: '13px',
    marginBottom: '12px',
  },
  success: {
    padding: '12px',
    background: 'rgba(80,200,120,0.2)',
    border: '1px solid rgba(80,200,120,0.3)',
    borderRadius: '6px',
    color: '#50c878',
    fontSize: '13px',
    marginBottom: '12px',
  },
  stats: {
    padding: '12px',
    background: 'rgba(255,255,255,0.05)',
    borderRadius: '6px',
    fontSize: '13px',
  },
  statRow: {
    display: 'flex',
    justifyContent: 'space-between',
    marginBottom: '6px',
  },
  statLabel: {
    color: 'rgba(255,255,255,0.6)',
  },
  statValue: {
    color: '#50c878',
    fontWeight: 600,
  },
  colorLegend: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '8px',
    marginTop: '12px',
  },
  colorItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    fontSize: '11px',
    color: 'rgba(255,255,255,0.7)',
  },
  colorDot: {
    width: '12px',
    height: '12px',
    borderRadius: '3px',
  },
  hiddenInput: {
    display: 'none',
  },
}

function ControlPanel({
  models,
  files,
  selectedModel,
  selectedSize,
  onModelChange,
  onSizeChange,
  onLoadFile,
  onAnalyze,
  onRefreshFiles,
  error,
  gridData,
}) {
  const [hoveredFile, setHoveredFile] = React.useState(null)
  const [uploading, setUploading] = React.useState(false)
  const [uploadStatus, setUploadStatus] = React.useState(null)
  const fileInputRef = React.useRef(null)

  const handleUploadClick = () => {
    fileInputRef.current?.click()
  }

  const handleFileSelect = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return

    if (!file.name.endsWith('.json')) {
      setUploadStatus({ type: 'error', message: 'Only JSON files are supported' })
      return
    }

    setUploading(true)
    setUploadStatus(null)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Upload failed')
      }

      const data = await res.json()
      setUploadStatus({ type: 'success', message: `Uploaded: ${data.filename}` })

      // Refresh file list
      onRefreshFiles()

      // Auto-load the uploaded file
      setTimeout(() => {
        onLoadFile(data.filepath)
      }, 500)
    } catch (err) {
      setUploadStatus({ type: 'error', message: err.message })
    } finally {
      setUploading(false)
      // Reset input so same file can be selected again
      e.target.value = ''
    }
  }

  // Clear upload status after 5 seconds
  React.useEffect(() => {
    if (uploadStatus) {
      const timer = setTimeout(() => setUploadStatus(null), 5000)
      return () => clearTimeout(timer)
    }
  }, [uploadStatus])

  const modelOptions = Object.keys(models)
  const sizeOptions = models[selectedModel]?.layer_counts
    ? Object.keys(models[selectedModel].layer_counts)
    : ['7b']

  const colors = {
    embedding: '#4a90d9',
    attention: '#50c878',
    feedforward: '#ff7f50',
    output: '#da70d6',
  }

  return (
    <div>
      {/* Error display */}
      {error && (
        <div style={styles.error}>
          {error}
        </div>
      )}

      {/* Upload status */}
      {uploadStatus && (
        <div style={uploadStatus.type === 'error' ? styles.error : styles.success}>
          {uploadStatus.message}
        </div>
      )}

      {/* Model Selection */}
      <div style={styles.section}>
        <div style={styles.sectionTitle}>Model Profile</div>
        <select
          style={styles.select}
          value={selectedModel}
          onChange={(e) => onModelChange(e.target.value)}
        >
          {modelOptions.map((model) => (
            <option key={model} value={model} style={styles.option}>
              {models[model]?.description || model}
            </option>
          ))}
        </select>
        <select
          style={styles.select}
          value={selectedSize}
          onChange={(e) => onSizeChange(e.target.value)}
        >
          {sizeOptions.map((size) => (
            <option key={size} value={size} style={styles.option}>
              {size.toUpperCase()}
            </option>
          ))}
        </select>
      </div>

      {/* File Upload */}
      <div style={styles.section}>
        <div style={styles.sectionTitle}>Upload Capture File</div>
        <input
          ref={fileInputRef}
          type="file"
          accept=".json"
          style={styles.hiddenInput}
          onChange={handleFileSelect}
        />
        <button
          style={styles.buttonUpload}
          onClick={handleUploadClick}
          disabled={uploading}
        >
          {uploading ? 'Uploading...' : 'Choose JSON File'}
        </button>
      </div>

      {/* File Selection */}
      <div style={styles.section}>
        <div style={styles.sectionTitle}>
          Capture Files
          <button
            style={{ ...styles.buttonSecondary, width: 'auto', marginLeft: '8px', padding: '4px 8px' }}
            onClick={onRefreshFiles}
          >
            Refresh
          </button>
        </div>
        <div style={styles.fileList}>
          {files.length === 0 ? (
            <div style={{ color: 'rgba(255,255,255,0.4)', fontSize: '13px' }}>
              No capture files found. Upload a JSON file above.
            </div>
          ) : (
            files.map((file) => (
              <div
                key={file.path}
                style={{
                  ...styles.fileItem,
                  ...(hoveredFile === file.path ? styles.fileItemHover : {}),
                }}
                onMouseEnter={() => setHoveredFile(file.path)}
                onMouseLeave={() => setHoveredFile(null)}
                onClick={() => onLoadFile(file.path)}
              >
                {file.name}
              </div>
            ))
          )}
        </div>
      </div>

      {/* Analyze Button */}
      <div style={styles.section}>
        <button
          style={styles.button}
          onClick={onAnalyze}
        >
          Analyze Data
        </button>
      </div>

      {/* Statistics */}
      {gridData && (
        <div style={styles.section}>
          <div style={styles.sectionTitle}>Statistics</div>
          <div style={styles.stats}>
            <div style={styles.statRow}>
              <span style={styles.statLabel}>Total Layers:</span>
              <span style={styles.statValue}>{gridData.total_layers}</span>
            </div>
            <div style={styles.statRow}>
              <span style={styles.statLabel}>Tensor Types:</span>
              <span style={styles.statValue}>{gridData.tensor_types?.length || 0}</span>
            </div>
            <div style={styles.statRow}>
              <span style={styles.statLabel}>Grid Points:</span>
              <span style={styles.statValue}>{gridData.grid?.length || 0}</span>
            </div>
            <div style={styles.statRow}>
              <span style={styles.statLabel}>Max Count:</span>
              <span style={styles.statValue}>{gridData.max_count || 0}</span>
            </div>
          </div>

          {/* Color Legend */}
          <div style={styles.colorLegend}>
            {Object.entries(colors).map(([name, color]) => (
              <div key={name} style={styles.colorItem}>
                <div style={{ ...styles.colorDot, background: color }} />
                {name}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Help text */}
      <div style={{ fontSize: '11px', color: 'rgba(255,255,255,0.4)', marginTop: '24px' }}>
        <p>Drag to rotate, scroll to zoom, right-click to pan.</p>
      </div>
    </div>
  )
}

export default ControlPanel
