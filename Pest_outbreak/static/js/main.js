// static/js/main.js

// Show/hide loading spinner
function showLoading() {
    if ($('#loadingSpinner').length === 0) {
        $('body').append(`
            <div id="loadingSpinner" class="spinner-overlay">
                <div class="spinner-container">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p class="mt-2">Processing...</p>
                </div>
            </div>
        `);
    }
    $('#loadingSpinner').fadeIn();
}

function hideLoading() {
    $('#loadingSpinner').fadeOut();
}

// Show toast notification
function showToast(message, type = 'success') {
    const toast = $(`
        <div class="toast align-items-center text-white bg-${type} border-0 position-fixed bottom-0 end-0 m-3" role="alert">
            <div class="d-flex">
                <div class="toast-body">${message}</div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `);
    
    $('body').append(toast);
    const bsToast = new bootstrap.Toast(toast[0]);
    bsToast.show();
    
    setTimeout(() => toast.remove(), 3000);
}

// Simulate sensor readings
function simulateSensor(sensor) {
    let value;
    switch(sensor) {
        case 'temperature':
            value = (20 + Math.random() * 20).toFixed(1);
            break;
        case 'humidity':
            value = (60 + Math.random() * 40).toFixed(1);
            break;
        case 'pressure':
            value = (980 + Math.random() * 50).toFixed(1);
            break;
        case 'light':
            value = Math.floor(10000 + Math.random() * 90000);
            break;
    }
    $(`#${sensor}`).val(value);
    showToast(`${sensor} reading: ${value}`, 'info');
}

// Validate sensor values
function validateSensorValues() {
    const temp = parseFloat($('#temperature').val()) || 0;
    const humidity = parseFloat($('#humidity').val()) || 0;
    const pressure = parseFloat($('#pressure').val()) || 0;
    const light = parseFloat($('#light').val()) || 0;
    
    let warnings = [];
    
    if (temp < 20 || temp > 40) {
        warnings.push('Temperature outside optimal range (20-40°C)');
    }
    if (humidity < 60 || humidity > 100) {
        warnings.push('Humidity outside optimal range (60-100%)');
    }
    if (pressure < 980 || pressure > 1030) {
        warnings.push('Pressure outside normal range (980-1030 hPa)');
    }
    
    if (warnings.length > 0) {
        $('#sensorWarnings').html(warnings.map(w => 
            `<div class="alert alert-warning py-2"><i class="fas fa-exclamation-triangle me-2"></i>${w}</div>`
        ).join(''));
    } else {
        $('#sensorWarnings').empty();
    }
}

// Initialize on document ready
$(document).ready(function() {
    // Auto-validate sensor inputs
    $('#temperature, #humidity, #pressure, #light').on('input', validateSensorValues);
    
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // Sidebar toggle memory
    $('#sidebarCollapse').on('click', function() {
        localStorage.setItem('sidebarCollapsed', $('#sidebar').hasClass('active'));
    });
    
    // Restore sidebar state
    if (localStorage.getItem('sidebarCollapsed') === 'true') {
        $('#sidebar').addClass('active');
    }
});