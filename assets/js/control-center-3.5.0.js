/* DEFENDR OS 3.5.0 — Portal/Admin UI enhancer */
(function () {
  'use strict';

  function ready(fn) {
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', fn, { once: true });
    } else {
      fn();
    }
  }

  function closestMatch(element, selectors) {
    if (!element || !element.closest) return null;
    return element.closest(selectors);
  }

  ready(function () {
    var path = String(window.location.pathname || '').toLowerCase();
    if (path.indexOf('/admin/') === -1 && path.indexOf('/portal/') === -1) return;

    document.body.classList.add('defendr-control-ui');
    document.body.classList.add(path.indexOf('/admin/') !== -1 ? 'defendr-admin-ui' : 'defendr-portal-ui');
    document.documentElement.setAttribute('data-defendr-ui', '3.5.0');

    /* Keep wide tables aligned and scrollable without changing their content. */
    document.querySelectorAll('table').forEach(function (table) {
      if (closestMatch(table, '.ui-table-wrap,.table-responsive,.table-wrap')) return;
      if (closestMatch(table, '.calendar,.schedule-grid,.invoice-document,.proposal-document')) return;
      var wrapper = document.createElement('div');
      wrapper.className = 'ui-table-wrap';
      table.parentNode.insertBefore(wrapper, table);
      wrapper.appendChild(table);
    });

    /* Normalize button-only controls for accessibility and visual consistency. */
    document.querySelectorAll('button, a.btn, [role="button"]').forEach(function (control) {
      if (!control.getAttribute('aria-label') && !String(control.textContent || '').trim()) {
        var title = control.getAttribute('title');
        if (title) control.setAttribute('aria-label', title);
      }
    });

    /* Add a mobile sidebar overlay while preserving existing menu behavior. */
    var sidebar = document.querySelector('.sidebar,.app-sidebar,.portal-sidebar,.admin-sidebar');
    if (sidebar && !document.querySelector('.ui-mobile-overlay')) {
      var overlay = document.createElement('div');
      overlay.className = 'ui-mobile-overlay';
      overlay.setAttribute('aria-hidden', 'true');
      document.body.appendChild(overlay);
      overlay.addEventListener('click', function () {
        document.body.classList.remove('ui-nav-open');
      });
    }

    var menuButtons = document.querySelectorAll('[data-sidebar-toggle],.sidebar-toggle,.menu-toggle,.mobile-menu-button,.nav-toggle');
    menuButtons.forEach(function (button) {
      button.addEventListener('click', function () {
        window.setTimeout(function () {
          var openByExistingClass = sidebar && (
            sidebar.classList.contains('open') ||
            sidebar.classList.contains('active') ||
            sidebar.classList.contains('show')
          );
          document.body.classList.toggle('ui-nav-open', openByExistingClass || !document.body.classList.contains('ui-nav-open'));
        }, 0);
      });
    });

    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape') document.body.classList.remove('ui-nav-open');
    });

    document.querySelectorAll('.sidebar a,.app-sidebar a,.portal-sidebar a,.admin-sidebar a').forEach(function (link) {
      link.addEventListener('click', function () {
        if (window.matchMedia('(max-width: 860px)').matches) {
          document.body.classList.remove('ui-nav-open');
        }
      });
    });

    /* Mark page structures for easier QA without changing application behavior. */
    document.querySelectorAll('.card,.panel,.widget,.dashboard-card,.stat-card,.metric-card').forEach(function (card) {
      card.setAttribute('data-ui-card', 'true');
    });

    window.dispatchEvent(new CustomEvent('defendr:ui-ready', { detail: { version: '3.5.0' } }));
  });
})();
