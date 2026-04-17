import { useEffect, useMemo, useRef, useState } from 'react'

interface FrameData {
  structural_drift_score: number
  relational_stability_score: number
  [key: string]: any
}

interface ReplayChartProps {
  frames: FrameData[]
  currentIndex: number
}

type YMode = 'auto' | 'normalized'

export default function ReplayChart({ frames, currentIndex }: ReplayChartProps) {
  const [yMode, setYMode] = useState<YMode>('auto')
  const scrollRef = useRef<HTMLDivElement | null>(null)

  const safeIndex = Math.max(0, Math.min(currentIndex, Math.max(frames.length - 1, 0)))

  const getDriftPercent = (frame: FrameData) => Math.max(0, Math.min(100, (frame.structural_drift_score || 0) * 100))
  const getInstabilityPercent = (frame: FrameData) => {
    const stability = Math.max(0, Math.min(100, (frame.relational_stability_score || 0) * 100))
    return 100 - stability
  }

  const driftValues = useMemo(() => frames.map((frame) => getDriftPercent(frame)), [frames])
  const instabilityValues = useMemo(() => frames.map((frame) => getInstabilityPercent(frame)), [frames])
  const visibleEndIndex = safeIndex
  const visibleDriftValues = driftValues.slice(0, visibleEndIndex + 1)
  const visibleInstabilityValues = instabilityValues.slice(0, visibleEndIndex + 1)

  const Y_MIN = 0
  const Y_MAX = 100

  const domain = useMemo(() => {
    if (yMode === 'normalized') return { min: Y_MIN, max: Y_MAX }

    const values = [...visibleDriftValues, ...visibleInstabilityValues]
    const dataMin = Math.min(...values)
    const dataMax = Math.max(...values)
    const spread = Math.max(dataMax - dataMin, 0)

    const padding = spread < 4
      ? Math.max(spread * 0.06, 0.15)
      : Math.max(spread * 0.04, 0.3)

    const min = Math.max(Y_MIN, dataMin - padding)
    const max = Math.min(Y_MAX, dataMax + padding)

    if (max - min < 1.2) {
      const center = (max + min) / 2
      return {
        min: Math.max(Y_MIN, center - 0.35),
        max: Math.min(Y_MAX, center + 0.35),
      }
    }

    return { min, max }
  }, [visibleDriftValues, visibleInstabilityValues, yMode])

  if (frames.length === 0) return null

  const H = 420
  const M = { top: 30, right: 30, bottom: 48, left: 66 }
  const pixelsPerFrame = 12
  const minPlotWidth = 920
  const plotWidth = Math.max(minPlotWidth, Math.max(frames.length - 1, 1) * pixelsPerFrame)
  const W = M.left + M.right + plotWidth
  const IH = H - M.top - M.bottom

  const yToPx = (value: number) => {
    const den = Math.max(domain.max - domain.min, 0.001)
    const t = (value - domain.min) / den
    return M.top + IH - t * IH
  }

  const xFor = (idx: number) => M.left + ((idx / Math.max(frames.length - 1, 1)) * plotWidth)

  const driftPoints = driftValues.map((value, idx) => ({ x: xFor(idx), y: yToPx(value), value }))
  const instabilityPoints = instabilityValues.map((value, idx) => ({ x: xFor(idx), y: yToPx(value), value }))
  const visibleDriftPoints = driftPoints.slice(0, visibleEndIndex + 1)
  const visibleInstabilityPoints = instabilityPoints.slice(0, visibleEndIndex + 1)

  const currentX = xFor(safeIndex)
  const activeBandWidth = Math.max(10, pixelsPerFrame * 0.85)
  const currentDriftPoint = driftPoints[safeIndex]
  const currentInstabilityPoint = instabilityPoints[safeIndex]

  const driftPath = visibleDriftPoints.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')
  const instabilityPath = visibleInstabilityPoints.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')

  const yTicks = Array.from({ length: 5 }).map((_, i) => {
    const t = i / 4
    const v = domain.min + (domain.max - domain.min) * t
    return { value: v, y: yToPx(v) }
  })
  const thresholdLines = [
    { label: 'Watch', value: 55, stroke: 'rgba(245, 158, 11, 0.5)' },
    { label: 'Critical', value: 78, stroke: 'rgba(239, 68, 68, 0.55)' },
  ]

  useEffect(() => {
    const el = scrollRef.current
    if (!el) return

    const targetLeft = currentX - el.clientWidth * 0.5
    const maxScroll = Math.max(el.scrollWidth - el.clientWidth, 0)
    el.scrollTo({
      left: Math.max(0, Math.min(targetLeft, maxScroll)),
      behavior: 'smooth',
    })
  }, [currentX])

  return (
    <div className="panel replay-chart-panel">
      <div className="panel-head replay-chart-head">
        <div>
          <span className="eyebrow">Replay Timeline</span>
          <span className="panel-subtitle">Live-feed replay of structural drift and relational instability across {frames.length} frames</span>
        </div>
        <div className="chart-head-values" aria-live="polite">
          <div><span>Drift</span><strong>{currentDriftPoint?.value.toFixed(2)}%</strong></div>
          <div><span>Instability</span><strong>{currentInstabilityPoint?.value.toFixed(2)}%</strong></div>
        </div>
      </div>

      <div className="chart-toolbar" role="group" aria-label="Y-axis scaling mode">
        <button
          className={`btn btn-subtle ${yMode === 'auto' ? 'active' : ''}`}
          onClick={() => setYMode('auto')}
          type="button"
        >
          Auto-fit data view
        </button>
        <button
          className={`btn btn-subtle ${yMode === 'normalized' ? 'active' : ''}`}
          onClick={() => setYMode('normalized')}
          type="button"
        >
          Normalized 0–100
        </button>
      </div>

      <div className="chart-legend" aria-hidden="true">
        <span><i className="swatch swatch-drift" /> Structural drift</span>
        <span><i className="swatch swatch-instability" /> Relational instability</span>
        <span><i className="swatch swatch-active" /> Active frame</span>
      </div>

      <div ref={scrollRef} className="chart-scroll-container">
        <div className="chart-container" style={{ minWidth: `${W}px` }}>
          <svg className="chart-svg" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
            <rect x={0} y={0} width={W} height={H} fill="rgba(5,10,24,0.72)" />
            <defs>
              <filter id="glow">
                <feGaussianBlur stdDeviation="2" result="coloredBlur" />
                <feMerge>
                  <feMergeNode in="coloredBlur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>

            {yTicks.map((tick) => (
              <g key={`grid-${tick.value.toFixed(4)}`}>
                <line
                  x1={M.left}
                  y1={tick.y}
                  x2={M.left + plotWidth}
                  y2={tick.y}
                  stroke="rgba(255,255,255,0.1)"
                  strokeWidth="1"
                />
                <text x={M.left - 10} y={tick.y + 4} fill="rgba(203,213,225,0.8)" fontSize="11" textAnchor="end">
                  {tick.value.toFixed(1)}
                </text>
              </g>
            ))}
            {thresholdLines
              .filter((line) => line.value >= domain.min && line.value <= domain.max)
              .map((line) => (
                <g key={line.label}>
                  <line
                    x1={M.left}
                    y1={yToPx(line.value)}
                    x2={M.left + plotWidth}
                    y2={yToPx(line.value)}
                    stroke={line.stroke}
                    strokeDasharray="6 4"
                    strokeWidth="1.3"
                  />
                  <text
                    x={M.left + 8}
                    y={yToPx(line.value) - 6}
                    fill={line.stroke}
                    fontSize="10"
                    fontWeight="700"
                  >
                    {line.label}
                  </text>
                </g>
              ))}

            <line x1={M.left} y1={M.top} x2={M.left} y2={M.top + IH} stroke="rgba(255,255,255,0.35)" strokeWidth="1.5" />
            <line x1={M.left} y1={M.top + IH} x2={M.left + plotWidth} y2={M.top + IH} stroke="rgba(255,255,255,0.35)" strokeWidth="1.5" />

            <rect
              x={currentX - activeBandWidth / 2}
              y={M.top}
              width={activeBandWidth}
              height={IH}
              fill="rgba(6,182,212,0.14)"
              rx="3"
            />

            <rect
              x={currentX + activeBandWidth / 2}
              y={M.top}
              width={Math.max((M.left + plotWidth) - (currentX + activeBandWidth / 2), 0)}
              height={IH}
              fill="rgba(2,6,23,0.5)"
            />

            <polyline points={instabilityPath} fill="none" stroke="#FF8A5C" strokeWidth="2.6" opacity="0.98" filter="url(#glow)" />
            <polyline points={driftPath} fill="none" stroke="#60A5FA" strokeWidth="2.8" opacity="1" filter="url(#glow)" />

            {visibleDriftPoints
              .filter((point) => point.value >= 78)
              .map((point, idx) => (
                <circle
                  key={`high-drift-${idx}`}
                  cx={point.x}
                  cy={point.y}
                  r="3.6"
                  fill="#ef4444"
                  stroke="#fee2e2"
                  strokeWidth="1.2"
                  opacity="0.9"
                />
              ))}

            <line
              x1={currentX}
              y1={M.top}
              x2={currentX}
              y2={M.top + IH}
              stroke="#22D3EE"
              strokeWidth="2.4"
              opacity="0.95"
            />

            {currentDriftPoint && (
              <circle cx={currentX} cy={currentDriftPoint.y} r="7.5" fill="#60A5FA" stroke="#FFFFFF" strokeWidth="2" opacity="1" />
            )}
            {currentInstabilityPoint && (
              <circle cx={currentX} cy={currentInstabilityPoint.y} r="7.5" fill="#FF8A5C" stroke="#FFFFFF" strokeWidth="2" opacity="1" />
            )}

          </svg>
        </div>
      </div>

      <div className="chart-footer">
        <span>Frame {safeIndex + 1} / {frames.length}</span>
      </div>
    </div>
  )
}
