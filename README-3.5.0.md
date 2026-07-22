# DEFENDR OS 3.5.0 — Unified Portal & Admin Design

This update standardizes the visual system used by the company portal and platform-owner admin.

## Improvements

- Consistent KLEVR-inspired navy, royal-blue and aqua palette
- Symmetrical dashboard grids and equal-height cards
- Unified spacing, borders, corner radii and shadows
- Standardized buttons, labels, form controls and helper text
- Clean responsive table containers
- Balanced page headers and action toolbars
- Cleaner navigation hierarchy and active states
- Mobile sidebar overlay and responsive one-column layouts
- Controlled icon dimensions
- Print-safe cards and reports

## Install over DEFENDR OS 3.4.2

1. Back up the current installation.
2. Upload the contents of this update folder to the DEFENDR OS application root.
3. Confirm that these files exist:
   - `assets/css/control-center-3.5.0.css`
   - `assets/js/control-center-3.5.0.js`
   - `apply-ui-3.5.0.php`
4. Visit `https://your-domain.example/apply-ui-3.5.0.php` once.
5. Review both the company portal and owner admin.
6. Remove `apply-ui-3.5.0.php` after successful verification.

The installer creates a dated rollback copy under `storage/backups/` before changing any shared header or JavaScript file.

## Scope

This is an upgrade-safe UI package. It does not modify company records, subscriptions, integrations, AI settings, permissions, billing data or operational workflows.
