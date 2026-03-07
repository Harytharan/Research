/* ============================================================================
   PADDYPREDICT — MAIN APPLICATION JAVASCRIPT
   ============================================================================ */

document.addEventListener('DOMContentLoaded', function () {

    // ===== SIDEBAR TOGGLE =====
    const sidebar = document.getElementById('sidebar');
    const toggleBtn = document.getElementById('sidebarToggle');
    let overlay = document.querySelector('.sidebar-overlay');

    // Create overlay if it doesn't exist
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.className = 'sidebar-overlay';
        document.body.appendChild(overlay);
    }

    if (toggleBtn && sidebar) {
        toggleBtn.addEventListener('click', function () {
            const isMobile = window.innerWidth <= 992;
            if (isMobile) {
                sidebar.classList.toggle('mobile-open');
                overlay.classList.toggle('show');
            } else {
                sidebar.classList.toggle('collapsed');
            }
        });
    }

    // Close sidebar on overlay click (mobile)
    overlay.addEventListener('click', function () {
        sidebar.classList.remove('mobile-open');
        overlay.classList.remove('show');
    });

    // Handle window resize
    window.addEventListener('resize', function () {
        if (window.innerWidth > 992) {
            sidebar.classList.remove('mobile-open');
            overlay.classList.remove('show');
        }
    });

    // ===== SUBMENU TOGGLE =====
    document.querySelectorAll('.sub-toggle').forEach(function (toggle) {
        toggle.addEventListener('click', function (e) {
            e.preventDefault();
            const parent = this.closest('.nav-item.has-sub');

            // Close other open submenus
            document.querySelectorAll('.nav-item.has-sub.open').forEach(function (item) {
                if (item !== parent) {
                    item.classList.remove('open');
                }
            });

            // Toggle this submenu
            parent.classList.toggle('open');
        });
    });

    // ===== AUTO-DISMISS FLASH MESSAGES =====
    setTimeout(function () {
        document.querySelectorAll('.flash-area .alert, .flash-container .alert').forEach(function (alert) {
            alert.style.transition = 'opacity 0.4s ease';
            alert.style.opacity = '0';
            setTimeout(function () { alert.remove(); }, 400);
        });
    }, 5000);

    // ===== SIDEBAR STATE PERSISTENCE =====
    const sidebarState = localStorage.getItem('sidebarCollapsed');
    if (sidebarState === 'true' && window.innerWidth > 992 && sidebar) {
        sidebar.classList.add('collapsed');
    }

    if (toggleBtn && sidebar) {
        toggleBtn.addEventListener('click', function () {
            if (window.innerWidth > 992) {
                localStorage.setItem('sidebarCollapsed', sidebar.classList.contains('collapsed'));
            }
        });
    }

    // ===== SMOOTH PAGE TRANSITIONS =====
    document.querySelectorAll('.page-content').forEach(function (content) {
        content.style.animation = 'fadeIn 0.4s ease';
    });

});