export default function LoadingSkeleton() {
  return (
    <div className="demo-app">
      <div className="demo-header skeleton-header">
        <div className="skeleton skeleton-title" style={{ width: '240px', height: '24px' }} />
        <div className="skeleton skeleton-metric" style={{ width: '320px', height: '32px' }} />
      </div>

      <div className="session-strip skeleton-session">
        <div className="session-strip-head">
          <div>
            <div className="skeleton" style={{ width: '160px', height: '16px', marginBottom: '8px' }} />
            <div className="skeleton" style={{ width: '240px', height: '14px' }} />
          </div>
          <div className="skeleton" style={{ width: '280px', height: '14px' }} />
        </div>
        <div className="session-progress-track skeleton-progress">
          <div className="skeleton-progress-bar" />
        </div>
        <div className="session-grid">
          {[1, 2, 3].map((i) => (
            <div key={i} className="session-card skeleton-card">
              <div className="skeleton" style={{ width: '80px', height: '12px', marginBottom: '8px' }} />
              <div className="skeleton" style={{ width: '120px', height: '24px', marginBottom: '8px' }} />
              <div className="skeleton" style={{ width: '100px', height: '12px' }} />
            </div>
          ))}
        </div>
      </div>

      <div className="demo-controls-row skeleton-controls" style={{ opacity: 0.5 }}>
        <div className="skeleton" style={{ width: '100%', maxWidth: '600px', height: '32px' }} />
      </div>

      <div className="demo-main">
        <div className="skeleton-panel" style={{ height: '300px', marginBottom: '28px' }} />
        <div className="skeleton-panel" style={{ height: '400px', marginBottom: '28px' }} />
        <div className="skeleton-panel" style={{ height: '280px' }} />
      </div>
    </div>
  )
}
