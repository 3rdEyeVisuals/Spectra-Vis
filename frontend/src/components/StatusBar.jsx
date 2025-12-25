/**
 * Spectra Vis - Status Bar Component
 * Copyright (c) 2025 3rdEyeVisuals
 */

import React from 'react'

const styles = {
  container: {
    display: 'flex',
    alignItems: 'center',
    gap: '16px',
  },
  status: {
    display: 'flex',
    alignItems: 'center',
    gap: '6px',
    fontSize: '12px',
  },
  dot: {
    width: '8px',
    height: '8px',
    borderRadius: '50%',
  },
  file: {
    fontSize: '12px',
    color: 'rgba(255,255,255,0.6)',
    maxWidth: '300px',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
}

const statusColors = {
  connected: '#50c878',
  checking: '#ffd700',
  disconnected: '#ff6b6b',
  error: '#ff6b6b',
}

const statusText = {
  connected: 'Connected',
  checking: 'Connecting...',
  disconnected: 'Disconnected',
  error: 'Error',
}

function StatusBar({ status, loadedFile }) {
  return (
    <div style={styles.container}>
      {loadedFile && (
        <div style={styles.file}>
          Loaded: {loadedFile.split(/[/\\]/).pop()}
        </div>
      )}
      <div style={styles.status}>
        <div
          style={{
            ...styles.dot,
            background: statusColors[status] || statusColors.error,
            boxShadow: `0 0 8px ${statusColors[status] || statusColors.error}`,
          }}
        />
        <span style={{ color: statusColors[status] }}>
          {statusText[status] || status}
        </span>
      </div>
    </div>
  )
}

export default StatusBar
