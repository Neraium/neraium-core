'use client'

import React from 'react'
import styles from './HistoricalTimeline.module.css'

interface HistoryPoint {
  timestamp: number
  temp: number
  humidity: number
  co2: number
  vpd: number
  ec: number
  ph: number
}

interface Props {
  history: HistoryPoint[]
  playbackTime: number | null
  onPlaybackChange: (time: number) => void
  isPlaying: boolean
  onPlayToggle: () => void
}

export default function HistoricalTimeline({
  history,
  playbackTime,
  onPlaybackChange,
  isPlaying,
  onPlayToggle,
}: Props) {
  const maxTime = history.length > 0 ? history[history.length - 1].timestamp : 0
  const minTime = history.length > 0 ? history[0].timestamp : 0

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onPlaybackChange(minTime + (parseInt(e.target.value) / 100) * (maxTime - minTime))
  }

  const formatTime = (timestamp: number) => {
    const minutes = Math.floor((timestamp - minTime) / 60000)
    const seconds = Math.floor(((timestamp - minTime) % 60000) / 1000)
    return `${minutes}m ${seconds}s`
  }

  const currentPercent =
    playbackTime !== null
      ? ((playbackTime - minTime) / (maxTime - minTime)) * 100
      : 100

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h2>Room State Evolution Timeline</h2>
        <div className={styles.controls}>
          <button
            className={styles.playButton}
            onClick={onPlayToggle}
            title={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? '⏸️' : '▶️'}
          </button>
          <span className={styles.time}>
            {formatTime(playbackTime || maxTime)}
          </span>
        </div>
      </div>

      <div className={styles.scrubber}>
        <input
          type="range"
          min="0"
          max="100"
          value={currentPercent}
          onChange={handleSliderChange}
          className={styles.slider}
        />
        <div className={styles.labels}>
          <span>Start</span>
          <span>Now</span>
        </div>
      </div>

      <div className={styles.hintText}>
        Drag to scrub through time and see when degradation started
      </div>
    </div>
  )
}
