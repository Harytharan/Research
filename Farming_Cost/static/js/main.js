// static/js/main.js

// Show loading spinner
function showLoading() {
    $('.spinner-overlay').fadeIn();
}

// Hide loading spinner
function hideLoading() {
    $('.spinner-overlay').fadeOut();
}

// Format number with commas
function formatNumber(num) {
    return num.toString().replace(/(\d)(?=(\d{3})+(?!\d))/g, '$1,');
}

// Show toast notification
function showToast(message, type = 'success') {
    const toast = $(`
        <div class="toast align-items-center text-white bg-${type} border-0 position-fixed bottom-0 end-0 m-3" role="alert">
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `);
    
    $('body').append(toast);
    const bsToast = new bootstrap.Toast(toast[0]);
    bsToast.show();
    
    setTimeout(() => {
        toast.remove();
    }, 3000);
}

// Calculate total of cost inputs
function calculateTotal() {
    let total = 0;
    $('input[name$="_cost"]').each(function() {
        total += parseFloat($(this).val()) || 0;
    });
    $('#totalCost').text('LKR ' + formatNumber(total));
}

// Auto-calculate total on input change
$(document).ready(function() {
    $('input[name$="_cost"]').on('input', calculateTotal);
    
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});

// Handle form submissions with loading
$('form').on('submit', function() {
    showLoading();
});

// Auto-dismiss alerts after 5 seconds
setTimeout(function() {
    $('.alert').fadeOut('slow');
}, 5000);

// Sidebar toggle memory
$('#sidebarCollapse').on('click', function() {
    localStorage.setItem('sidebarCollapsed', $('#sidebar').hasClass('active'));
});

// Restore sidebar state
if (localStorage.getItem('sidebarCollapsed') === 'true') {
    $('#sidebar').addClass('active');
}

// Validate sensor values
function validateSensorValues(temp, humidity) {
    let warnings = [];
    
    if (temp < 15 || temp > 45) {
        warnings.push('Temperature is outside typical range (15-45°C)');
    }
    if (humidity < 40 || humidity > 100) {
        warnings.push('Humidity is outside typical range (40-100%)');
    }
    
    return warnings;
}

// Update sensor warnings
$('#temperature, #humidity').on('input', function() {
    const temp = parseFloat($('#temperature').val()) || 0;
    const humidity = parseFloat($('#humidity').val()) || 0;
    
    const warnings = validateSensorValues(temp, humidity);
    
    if (warnings.length > 0) {
        $('#sensorWarnings').html(warnings.map(w => 
            `<div class="text-warning"><i class="fas fa-exclamation-triangle me-2"></i>${w}</div>`
        ).join(''));
    } else {
        $('#sensorWarnings').empty();
    }
});

// Export results as PDF
function exportToPDF() {
    window.print();
}

// Copy results to clipboard
$('#copyResults').on('click', function() {
    const results = $('#resultsTable').text();
    navigator.clipboard.writeText(results).then(function() {
        showToast('Results copied to clipboard!', 'success');
    });
});