// Add these to your existing main.js

// Show/hide loading spinner
function showLoading() {
    if ($('#loadingSpinner').length === 0) {
        $('body').append(`
            <div id="loadingSpinner" class="spinner-overlay">
                <div class="spinner-container">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                    <p class="mt-2">Analyzing field data...</p>
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

// Format currency
function formatCurrency(amount) {
    return 'LKR ' + amount.toFixed(2).replace(/\d(?=(\d{3})+\.)/g, '$&,');
}

// Validate numeric inputs
function validateNumber(input, min, max) {
    const val = parseFloat(input.value);
    if (isNaN(val) || val < min || val > max) {
        input.classList.add('is-invalid');
        return false;
    } else {
        input.classList.remove('is-invalid');
        return true;
    }
}

// Auto-calculate totals (if needed)
$(document).ready(function() {
    // Add input validation listeners
    $('input[type="number"]').on('input', function() {
        const min = parseFloat($(this).attr('min')) || 0;
        const max = parseFloat($(this).attr('max')) || 100;
        validateNumber(this, min, max);
    });
    
    // Form submission loading
    $('form').on('submit', function() {
        showLoading();
    });
    
    // Auto-dismiss alerts
    setTimeout(function() {
        $('.alert-dismissible').fadeOut('slow');
    }, 5000);
    
    // Sidebar state memory
    if (localStorage.getItem('sidebarCollapsed') === 'true') {
        $('#sidebar').addClass('active');
    }
    
    $('#sidebarCollapse').on('click', function() {
        localStorage.setItem('sidebarCollapsed', $('#sidebar').hasClass('active'));
    });
});