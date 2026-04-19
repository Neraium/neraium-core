// Geometry module for 3D visualization

export function initGeometry() {
    const viewport = document.getElementById('geometryViewport');
    if (viewport) {
        viewport.innerHTML = '<div style="padding: 20px; text-align: center;">Structural geometry visualization initialized</div>';
    }
}

export function renderGeometry(data) {
    // Render structural geometry data - tetrahedron with drift/stability metrics
    if (!data) return;

    const viewport = document.getElementById('geometryViewport');
    if (!viewport) return;

    // Extract metrics from data
    const drift = typeof data.structural_drift_score === 'number' ? data.structural_drift_score : 0;
    const stability = typeof data.stability === 'number' ? data.stability : 0;
    const coherence = typeof data.coherence === 'number' ? data.coherence : 0;
    const confidence = typeof data.confidence === 'number' ? data.confidence : 0;
    const composite = typeof data.latest_instability === 'number' ? data.latest_instability : 0;

    // Render tetrahedron structure with metrics
    const html = `
        <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; gap: 20px; padding: 20px;">
            <div style="display: flex; align-items: center; justify-content: center; gap: 30px; flex-wrap: wrap;">
                <div style="text-align: center; min-width: 80px;">
                    <div style="font-size: 24px; font-weight: bold; color: #79abff;">${drift.toFixed(1)}</div>
                    <div style="font-size: 12px; color: #999;">Drift</div>
                </div>
                <div style="text-align: center; min-width: 80px;">
                    <div style="font-size: 24px; font-weight: bold; color: #88dd00;">${stability.toFixed(1)}</div>
                    <div style="font-size: 12px; color: #999;">Stability</div>
                </div>
                <div style="text-align: center; min-width: 80px;">
                    <div style="font-size: 24px; font-weight: bold; color: #1dd1a1;">${coherence.toFixed(1)}</div>
                    <div style="font-size: 12px; color: #999;">Coherence</div>
                </div>
                <div style="text-align: center; min-width: 80px;">
                    <div style="font-size: 24px; font-weight: bold; color: #ff8c00;">${confidence.toFixed(1)}</div>
                    <div style="font-size: 12px; color: #999;">Confidence</div>
                </div>
            </div>
            <div style="font-size: 14px; color: #999;">Composite instability: ${composite.toFixed(2)}</div>
        </div>
    `;

    viewport.innerHTML = html;
}

export function updateGeometryView(data) {
    // Update the geometry visualization based on new data
    renderGeometry(data);
}
