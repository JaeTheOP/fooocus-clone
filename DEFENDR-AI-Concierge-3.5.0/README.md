# DEFENDR OS 3.5.0 — AI Website Concierge

This upgrade installer adds **Nova**, a public AI Growth Guide to the DEFENDR OS homepage.

## Included

- Floating AI character on `index.php`
- Automatic, approximate city/region/country greeting
- Cloudflare geolocation-header support
- Optional IP geolocation fallback when Cloudflare headers are unavailable
- No raw IP storage and no raw IP sent to OpenRouter
- OpenRouter-backed product conversations
- Deterministic feature answers when the AI provider is unavailable
- Quick prompts, plan guidance, demo handoff, and trial handoff
- Optional browser speech button
- Per-session daily message limits
- Owner settings page at `/admin/ai-concierge.php`
- Automatic backup before changing application files

## Install

1. Back up DEFENDR OS.
2. Upload `install-ai-concierge-3.5.0.php` beside the main `index.php` file.
3. Visit the installer in a browser.
4. Click **Install AI Concierge 3.5.0**.
5. Delete the installer after success.
6. Clear browser, CDN, and server caches.
7. Open `/admin/ai-concierge.php` to configure Nova.

## OpenRouter

The concierge attempts to use the OpenRouter provider already configured under the DEFENDR OS owner admin. The API key remains server-side. If the provider cannot be loaded, the concierge still answers common product questions through its built-in feature guide.

## Privacy

The concierge uses coarse location context only. It does not request precise browser geolocation. It does not send a visitor's raw IP address to OpenRouter. Fallback IP lookups may be sent to the configured public geolocation service solely to obtain city/region/country data. Visitors with Do Not Track enabled receive a generic greeting.

## Files installed

- `app/ai_concierge_widget.php`
- `ai-concierge.php`
- `assets/css/ai-concierge-3.5.0.css`
- `assets/js/ai-concierge-3.5.0.js`
- `admin/ai-concierge.php`

The installer also adds one guarded include to `index.php`, updates `VERSION`, appends the changelog, and tries to add an admin navigation link without failing the installation when the navigation template differs.
