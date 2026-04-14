// Dashboard module

export function createToast(message) {
    const toastContainer = document.getElementById('toastContainer') || createToastContainer();
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    toastContainer.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}

function createToastContainer() {
    const container = document.createElement('div');
    container.id = 'toastContainer';
    container.style.position = 'fixed';
    container.style.top = '10px';
    container.style.right = '10px';
    container.style.zIndex = '9999';
    document.body.appendChild(container);
    return container;
}

export function riskBadgeHtml(riskLevel) {
    const colors = {
        critical: '#ff4444',
        high: '#ff8c00',
        medium: '#ffcc00',
        low: '#88dd00',
        none: '#00aa00'
    };
    const color = colors[riskLevel] || colors.none;
    return `<span class="risk-badge" style="background-color: ${color}; color: white; padding: 4px 8px; border-radius: 3px;">${riskLevel}</span>`;
}

export function phaseBadgeHtml(phase) {
    const colors = {
        baseline: '#4dabf7',
        learning: '#9c36b5',
        operation: '#15aabf',
        failure: '#fa5252'
    };
    const color = colors[phase] || '#666';
    return `<span class="phase-badge" style="background-color: ${color}; color: white; padding: 4px 8px; border-radius: 3px;">${phase}</span>`;
}

// Dashboard initialization
export function initDashboard() {
    const seedDemoBtn = document.getElementById('seedDemoBtn');
    const runsSearchInput = document.getElementById('runsSearchInput');
    const runResultsSearchInput = document.getElementById('runResultsSearchInput');
    const runRangeControls = document.getElementById('runRangeControls');
    const uploadDropZone = document.getElementById('uploadDropZone');
    const dashboardEmpty = document.getElementById('dashboardEmpty');
    const runDetailEmpty = document.getElementById('runDetailEmpty');
    const runResultsEmpty = document.getElementById('runResultsEmpty');

    if (seedDemoBtn) {
        seedDemoBtn.addEventListener('click', () => {
            createToast('Demo data seeded');
        });
    }
}
