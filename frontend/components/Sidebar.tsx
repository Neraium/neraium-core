"use client";

export function Sidebar() {
  return (
    <aside className="sidebar" aria-label="Main navigation">
      <div className="brand">
        <div className="brand-mark" aria-hidden="true">N</div>
        <div className="brand-text">
          <h1>Neraium</h1>
          <p>Synthesis Demo</p>
        </div>
      </div>
      <nav className="nav">
        <p className="nav-section-title">Navigation</p>
        <a href="/" className="active">Demo Dashboard</a>
      </nav>
    </aside>
  );
}
