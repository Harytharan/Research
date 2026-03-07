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

// Validate NPK values
function validateNPK(n, p, k) {
    let warnings = [];
    
    if (n < 400 || n > 650) {
        warnings.push('Nitrogen value is outside typical range (400-650 mg/kg)');
    }
    if (p < 3 || p > 20) {
        warnings.push('Phosphorus value is outside typical range (3-20 mg/kg)');
    }
    if (k < 3 || k > 20) {
        warnings.push('Potassium value is outside typical range (3-20 mg/kg)');
    }
    
    return warnings;
}

// Update prediction summary
function updateSummary(results) {
    $('#avgPrice').text(`₨ ${results.avg_price.toFixed(2)}`);
    $('#avgDemand').text(`${results.avg_demand.toFixed(0)} Tons`);
    $('#priceRange').text(`₨ ${results.price_range[0].toFixed(2)} - ₨ ${results.price_range[1].toFixed(2)}`);
    $('#demandRange').text(`${results.demand_range[0].toFixed(0)} - ${results.demand_range[1].toFixed(0)} Tons`);
}

// Export table to CSV
function exportToCSV(data, filename) {
    const csvContent = "data:text/csv;charset=utf-8," 
        + data.map(row => Object.values(row).join(",")).join("\n");
    
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", filename);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

// Initialize tooltips
$(document).ready(function() {
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

// Live sensor input validation
$('#nitrogen, #phosphorus, #potassium').on('input', function() {
    const n = parseFloat($('#nitrogen').val()) || 0;
    const p = parseFloat($('#phosphorus').val()) || 0;
    const k = parseFloat($('#potassium').val()) || 0;
    
    const warnings = validateNPK(n, p, k);
    
    if (warnings.length > 0) {
        $('#sensorWarnings').html(warnings.map(w => `<div class="text-warning"><i class="fas fa-exclamation-triangle me-2"></i>${w}</div>`).join(''));
    } else {
        $('#sensorWarnings').empty();
    }
});

// Date picker enhancement
$('input[type="date"]').on('focus', function() {
    this.showPicker();
});

// Copy prediction results to clipboard
$('#copyResults').on('click', function() {
    const results = $('#resultsTable').text();
    navigator.clipboard.writeText(results).then(function() {
        showToast('Results copied to clipboard!', 'success');
    });
});

// Print results
$('#printResults').on('click', function() {
    window.print();
});

// Download prediction as image
$('#downloadChart').on('click', function() {
    const canvas = document.getElementById('predictionChart');
    const link = document.createElement('a');
    link.download = 'prediction_chart.png';
    link.href = canvas.toDataURL();
    link.click();
});

// API call for quick prediction
$('#quickPredictBtn').on('click', async function() {
    const nDays = $('#quickDays').val();
    
    showLoading();
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ n_days: nDays })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showToast('Prediction completed successfully!', 'success');
            updateSummary(data.summary);
        } else {
            showToast('Error: ' + data.error, 'danger');
        }
    } catch (error) {
        showToast('Error making prediction', 'danger');
    } finally {
        hideLoading();
    }
});