/**
 * NeuralPlayground - Global JavaScript Utilities
 */

// ==========================================
// Navigation
// ==========================================

document.addEventListener('DOMContentLoaded', function () {
    // Mobile menu toggle
    const navbarToggle = document.getElementById('navbarToggle');
    const navbarMenu = document.getElementById('navbarMenu');

    if (navbarToggle && navbarMenu) {
        navbarToggle.addEventListener('click', function () {
            navbarMenu.classList.toggle('open');
        });

        // Handle dropdown in mobile
        const dropdowns = document.querySelectorAll('.navbar-dropdown');
        dropdowns.forEach(dropdown => {
            const toggle = dropdown.querySelector('.dropdown-toggle');
            if (toggle) {
                toggle.addEventListener('click', function (e) {
                    if (window.innerWidth <= 768) {
                        e.preventDefault();
                        dropdown.classList.toggle('open');
                    }
                });
            }
        });
    }

    // Navbar scroll effect
    const navbar = document.getElementById('navbar');
    if (navbar) {
        window.addEventListener('scroll', function () {
            if (window.scrollY > 10) {
                navbar.classList.add('scrolled');
            } else {
                navbar.classList.remove('scrolled');
            }
        });
    }
});

// ==========================================
// API Helpers
// ==========================================

/**
 * Make a POST request with JSON body
 */
async function postJSON(url, data) {
    const response = await fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    });

    const result = await response.json();

    if (!response.ok) {
        throw new Error(result.error || 'Request failed');
    }

    return result;
}

/**
 * Make a POST request with FormData (for file uploads)
 */
async function postFormData(url, formData) {
    const response = await fetch(url, {
        method: 'POST',
        body: formData
    });

    const result = await response.json();

    if (!response.ok) {
        throw new Error(result.error || 'Request failed');
    }

    return result;
}

// ==========================================
// UI Helpers
// ==========================================

/**
 * Show loading state on a container
 */
function showLoading(container, message = 'Processing...') {
    const overlay = document.createElement('div');
    overlay.className = 'loading-overlay';
    overlay.id = 'loading-overlay';
    overlay.innerHTML = `
        <div class="spinner"></div>
        <div class="loading-text">${message}</div>
    `;
    container.style.position = 'relative';
    container.appendChild(overlay);
}

/**
 * Hide loading state
 */
function hideLoading(container) {
    const overlay = container.querySelector('#loading-overlay');
    if (overlay) {
        overlay.remove();
    }
}

/**
 * Show error message
 */
function showError(container, message) {
    hideError(container);
    const alert = document.createElement('div');
    alert.className = 'alert alert-error';
    alert.id = 'error-alert';
    alert.innerHTML = `
        <span class="alert-icon">⚠️</span>
        <span>${message}</span>
    `;
    container.insertBefore(alert, container.firstChild);
}

/**
 * Hide error message
 */
function hideError(container) {
    const alert = container.querySelector('#error-alert');
    if (alert) {
        alert.remove();
    }
}

/**
 * Show success message
 */
function showSuccess(container, message) {
    hideSuccess(container);
    const alert = document.createElement('div');
    alert.className = 'alert alert-success';
    alert.id = 'success-alert';
    alert.innerHTML = `
        <span class="alert-icon">✅</span>
        <span>${message}</span>
    `;
    container.insertBefore(alert, container.firstChild);

    // Auto-hide after 5 seconds
    setTimeout(() => hideSuccess(container), 5000);
}

/**
 * Hide success message
 */
function hideSuccess(container) {
    const alert = container.querySelector('#success-alert');
    if (alert) {
        alert.remove();
    }
}

// ==========================================
// File Upload Helpers
// ==========================================

/**
 * Setup drag and drop for file upload
 */
function setupFileUpload(uploadElement, onFileSelect) {
    const input = uploadElement.querySelector('input[type="file"]');

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadElement.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Highlight on drag
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadElement.addEventListener(eventName, () => {
            uploadElement.classList.add('dragging');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadElement.addEventListener(eventName, () => {
            uploadElement.classList.remove('dragging');
        }, false);
    });

    // Handle drop
    uploadElement.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            onFileSelect(files[0]);
        }
    }, false);

    // Handle file input change
    if (input) {
        input.addEventListener('change', () => {
            if (input.files.length > 0) {
                onFileSelect(input.files[0]);
            }
        });
    }
}

/**
 * Validate file size
 */
function validateFileSize(file, maxSizeMB = 5) {
    const maxSize = maxSizeMB * 1024 * 1024;
    if (file.size > maxSize) {
        throw new Error(`File is too large. Maximum size is ${maxSizeMB}MB.`);
    }
    return true;
}

/**
 * Validate file type
 */
function validateFileType(file, allowedTypes) {
    const ext = file.name.split('.').pop().toLowerCase();
    if (!allowedTypes.includes(ext)) {
        throw new Error(`Invalid file type. Allowed types: ${allowedTypes.join(', ')}`);
    }
    return true;
}

// ==========================================
// Data Table Helpers
// ==========================================

/**
 * Create a data table from rows
 */
function createDataTable(fields, fieldLabels, rows, containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    let html = '<div class="data-table-wrapper"><table class="data-table">';

    // Header
    html += '<thead><tr>';
    fields.forEach(field => {
        html += `<th>${fieldLabels[field] || field}</th>`;
    });
    html += '</tr></thead>';

    // Body
    html += '<tbody>';
    rows.forEach(row => {
        html += '<tr>';
        fields.forEach(field => {
            html += `<td>${escapeHtml(row[field] || '')}</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody></table></div>';

    container.innerHTML = html;
}

/**
 * Escape HTML special characters
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ==========================================
// Download Helpers
// ==========================================

/**
 * Download text as a file
 */
function downloadText(text, filename, mimeType = 'text/plain') {
    const blob = new Blob([text], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

/**
 * Download data URI as file
 */
function downloadDataUri(dataUri, filename) {
    const a = document.createElement('a');
    a.href = dataUri;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

// ==========================================
// Form Helpers
// ==========================================

/**
 * Get form data as object
 */
function getFormData(formElement) {
    const formData = new FormData(formElement);
    const data = {};
    for (let [key, value] of formData.entries()) {
        data[key] = value;
    }
    return data;
}

/**
 * Disable/enable form elements during processing
 */
function setFormEnabled(formElement, enabled) {
    const elements = formElement.querySelectorAll('input, select, textarea, button');
    elements.forEach(el => {
        el.disabled = !enabled;
    });
}

// ==========================================
// Misc Utilities
// ==========================================

/**
 * Debounce function calls
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Copy text to clipboard
 */
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        return true;
    } catch (err) {
        console.error('Failed to copy:', err);
        return false;
    }
}
