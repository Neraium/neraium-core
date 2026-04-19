'use client'

import React from 'react'
import styles from './YieldImpact.module.css'

interface Props {
  yieldLoss: number
  daysInCycle: number
}

export default function YieldImpact({ yieldLoss, daysInCycle }: Props) {
  const getRecoverability = (loss: number): string => {
    if (loss < 5) return '100% recoverable'
    if (loss < 15) return '95% recoverable'
    if (loss < 25) return '80% recoverable'
    if (loss < 40) return '50% recoverable'
    return 'Significant damage'
  }

  return (
    <div className={styles.card}>
      <div className={styles.label}>Yield Impact</div>

      <div className={styles.gauge}>
        <div className={styles.gaugeBackground}>
          <div
            className={styles.gaugeFill}
            style={{
              width: `${Math.min(100, yieldLoss)}%`,
              backgroundColor:
                yieldLoss < 10
                  ? '#10b981'
                  : yieldLoss < 20
                    ? '#f59e0b'
                    : '#ef4444',
            }}
          />
        </div>
      </div>

      <div className={styles.loss}>{yieldLoss.toFixed(1)}% loss</div>
      <div className={styles.recoverable}>{getRecoverability(yieldLoss)}</div>
    </div>
  )
}
