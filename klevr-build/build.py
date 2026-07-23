from pathlib import Path
import hashlib
import json
import shutil
import zipfile

root = Path(__file__).resolve().parent
source = root / 'package'
dist = root / 'dist'
if dist.exists():
    shutil.rmtree(dist)
dist.mkdir(parents=True)

checksums = {}
for path in sorted(source.rglob('*')):
    if path.is_file():
        rel = path.relative_to(source).as_posix()
        checksums[rel] = hashlib.sha256(path.read_bytes()).hexdigest()

manifest = {
    'schema_version': 1,
    'package_id': 'klevr-equipment-price-reduction',
    'version': '2.0.0',
    'minimum_version': '1.9.9',
    'description': 'Reduce all current customer-facing equipment prices by 10 percent while leaving monitoring, installation, shipping, tax, and service fees unchanged.',
    'payload_path': 'payload',
    'requirements': {
        'php': '7.4.0',
        'extensions': ['pdo', 'zip', 'openssl'],
    },
    'delete_paths': [],
    'migrations': ['migrations/20260723_003_equipment_price_reduction.php'],
    'health_checks': ['health/release-2.0.0.php'],
    'checksums': checksums,
    'built_at': '2026-07-23T18:00:00Z',
}
(source / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')

package_name = 'klevr-software-update-2.0.0-equipment-price-reduction.zip'
package_path = dist / package_name
with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for path in sorted(source.rglob('*')):
        if path.is_file():
            archive.write(path, path.relative_to(source).as_posix())

sha = hashlib.sha256(package_path.read_bytes()).hexdigest()
(dist / 'klevr-software-update-2.0.0-equipment-price-reduction.sha256').write_text(
    f'{sha}  {package_name}\n', encoding='utf-8'
)

report = f'''KLEVR 2.0.0 — 10% EQUIPMENT PRICE REDUCTION
=================================================

Package: {package_name}
Minimum version: 1.9.9
Target version: 2.0.0
Database migrations: 1
Deleted files: 0
SHA-256: {sha}

SCOPE
-----
- Reduces all catalog-controlled AJAX customer equipment prices by 10%.
- Reduces current hardware add-on storefront prices by 10% when available.
- Leaves internal equipment cost unchanged.
- Leaves monitoring, installation, programming, shipping, tax, payment, and service fees unchanged.
- Does not alter historical orders, completed payments, subscriptions, or monitoring accounts.

UPDATED CORE EQUIPMENT PRICES
-----------------------------
- KLEVR Essential: $368.40 -> $331.56
- KLEVR Signature: $632.25 -> $569.03
- KLEVR RV: $397.08 -> $357.37
- KLEVR Apartment: $368.71 -> $331.84

UPDATED EXAMPLE ADD-ONS
-----------------------
- AJAX MotionProtect: $61.72 -> $55.55
- AJAX DoorProtect: $28.37 -> $25.53
- AJAX SpaceControl: $28.37 -> $25.53
- AJAX GlassProtect: $55.06 -> $49.55
- AJAX Doorbell: $233.94 -> $210.55
- Optional 2K WireFree Camera: $49.99 -> $44.99 when stored in hardware_addons

MARGIN EFFECT
-------------
The original Tier 2 customer price represented a 30% markup on cost. Reducing that customer price by 10% results in a 17% markup on cost and approximately a 14.53% equipment gross margin as a percentage of the new equipment price, before processing, fulfillment, warranty, returns, and overhead.

SAFETY
------
- Creates a protected backup of the original AJAX pricing catalog before modifying it.
- Uses a marker to prevent the reduction from being applied twice.
- Monitoring and installation prices are excluded.
- A live production installation or transaction was not performed during packaging.
'''
(dist / 'klevr-2.0.0-implementation-report.txt').write_text(report, encoding='utf-8')
print(sha)
