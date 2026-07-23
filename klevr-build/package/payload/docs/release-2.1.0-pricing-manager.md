# KLEVR 2.1.0 Central Pricing Manager

## Purpose

Add one administrator workspace for changing the package, plan, service, installation, and equipment prices shown to customers.

## Admin experience

- Adds **Admin > Pricing Manager**.
- Displays package and service prices from recognized database fields.
- Displays all customer prices from the shared AJAX Tier 2 catalog.
- Displays database-backed cameras, products, and hardware add-ons.
- Supports search by name, SKU, slug, type, and category.
- Shows internal equipment cost, customer price, gross margin, and installation reference where available.
- Updates margin calculations immediately in the browser before saving.

## Customer-facing behavior

The shared AJAX catalog already supplies Packages, System Builder, cart, checkout, and new order calculations. Changes made through Pricing Manager therefore flow to those customer-facing experiences without creating a second pricing source. Database-backed package, monitoring, installation, service, and add-on fields are updated in their existing records.

## Safety

- Creates a timestamped backup before replacing the AJAX catalog.
- Writes the catalog atomically.
- Uses prepared statements and an allowlist for database tables and fields.
- Validates prices between $0.00 and $1,000,000.00.
- Records successful and failed attempts in `storage/pricing-backups/pricing-manager-audit.jsonl`.
- Retains the latest 25 catalog backups.
- Does not reprice completed orders, payments, subscriptions, or active monitoring accounts.

## Installation

1. Open **Admin > Software Updates**.
2. Upload `klevr-software-update-2.1.0-pricing-manager.zip`.
3. Confirm the current version is at least 1.9.9 and the target version is 2.1.0.
4. Choose **Back Up & Apply**.
5. Confirm all health checks pass.
6. Open **Admin > Pricing Manager**.
7. Change one package or equipment price and save.
8. Verify the same price on Packages, System Builder, cart, and checkout.

A live production installation or payment transaction is not performed during packaging.
