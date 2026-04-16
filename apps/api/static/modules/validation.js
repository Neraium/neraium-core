// Validation module - applies analysis to replay data

export function initValidation() {
    const validationContainer = document.getElementById('runResultsSearchInput');
    if (validationContainer) {
        validationContainer.addEventListener('input', (e) => {
            const query = e.target.value;
            filterValidationResults(query);
        });
    }
}

export function filterValidationResults(query) {
    // Filter validation results by query text
    if (!query || !query.trim()) return;
    const results = document.querySelectorAll('[data-validation-result]');
    results.forEach(el => {
        const text = el.textContent.toLowerCase();
        const matches = text.includes(query.toLowerCase());
        el.style.display = matches ? '' : 'none';
    });
}

export function validateData(results) {
    // Apply validation/analysis logic to structural data
    if (!Array.isArray(results)) return { valid: false, errors: ['No data provided'] };

    const errors = [];
    const processedResults = results.map((r, idx) => {
        // Validate required fields
        if (!r.timestamp && !r.persisted_at) {
            errors.push(`Result ${idx}: Missing timestamp`);
        }
        if (typeof r.structural_drift_score !== 'number') {
            r.structural_drift_score = 0;
        }
        if (typeof r.latest_instability !== 'number') {
            r.latest_instability = 0;
        }
        // Ensure risk level is valid
        if (!['LOW', 'MEDIUM', 'HIGH', 'UNKNOWN'].includes(String(r.risk_level).toUpperCase())) {
            r.risk_level = 'UNKNOWN';
        }
        return r;
    });

    return {
        valid: errors.length === 0,
        errors: errors,
        results: processedResults
    };
}

export function displayValidationErrors(errors) {
    if (errors.length === 0) {
        return;
    }

    const errorContainer = document.createElement('div');
    errorContainer.className = 'validation-errors';
    errorContainer.innerHTML = '<h4>Validation Issues:</h4><ul>' +
        errors.map(e => `<li>${escapeHtml(String(e))}</li>`).join('') +
        '</ul>';

    const panel = document.querySelector('.panel');
    if (panel) {
        panel.appendChild(errorContainer);
    }
}

function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
    };
    return String(text).replace(/[&<>"']/g, m => map[m]);
}
