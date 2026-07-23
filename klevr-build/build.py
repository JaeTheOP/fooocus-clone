from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "package"
DIST = ROOT / "dist"
VERSION = "2.1.0"
PACKAGE_ID = "klevr-central-pricing-manager"
PACKAGE_NAME = f"klevr-software-update-{VERSION}-pricing-manager.zip"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_php() -> None:
    php_files = sorted(SOURCE.rglob("*.php"))
    for path in php_files:
        subprocess.run(["php", "-l", str(path)], check=True, capture_output=True, text=True)


def build_manifest() -> dict:
    checksums: dict[str, str] = {}
    for path in sorted(SOURCE.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        checksums[path.relative_to(SOURCE).as_posix()] = sha256(path)

    manifest = {
        "schema_version": 1,
        "package_id": PACKAGE_ID,
        "version": VERSION,
        "minimum_version": "1.9.9",
        "description": "Add a central administrator Pricing Manager for package, plan, service, installation, and customer equipment prices.",
        "payload_path": "payload",
        "requirements": {
            "php": "7.4.0",
            "extensions": ["pdo", "zip", "openssl"],
        },
        "delete_paths": [],
        "migrations": [
            "migrations/20260723_003_equipment_price_reduction.php",
            "migrations/20260723_004_pricing_manager.php",
        ],
        "health_checks": ["health/release-2.1.0.php"],
        "checksums": checksums,
        "built_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    (SOURCE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_archive() -> tuple[Path, str]:
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)

    package_path = DIST / PACKAGE_NAME
    with zipfile.ZipFile(package_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(SOURCE.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(SOURCE).as_posix())

    with zipfile.ZipFile(package_path, "r") as archive:
        bad_file = archive.testzip()
        if bad_file is not None:
            raise RuntimeError(f"ZIP integrity failed at {bad_file}")

    digest = sha256(package_path)
    (DIST / f"{PACKAGE_NAME}.sha256").write_text(f"{digest}  {PACKAGE_NAME}\n", encoding="utf-8")
    return package_path, digest


def write_report(package_path: Path, digest: str, manifest: dict) -> None:
    payload_files = [p for p in (SOURCE / "payload").rglob("*") if p.is_file()]
    report = f"""KLEVR 2.1.0 — CENTRAL PRICING MANAGER
=======================================

Package: {package_path.name}
Minimum version: {manifest['minimum_version']}
Target version: {manifest['version']}
Payload files: {len(payload_files)}
Database migrations: {len(manifest['migrations'])}
Deleted files: {len(manifest['delete_paths'])}
SHA-256: {digest}

IMPLEMENTED
-----------
- Adds Admin > Pricing Manager.
- Lets authorized administrators edit package, monitoring, installation, service, and equipment prices.
- Edits the shared AJAX Tier 2 customer-price catalog used by Packages, System Builder, cart, checkout, and new order calculations.
- Edits recognized price fields in products, packages, monitoring_plans, and hardware_addons when those tables exist.
- Shows internal equipment cost, customer price, gross margin, and installation reference.
- Provides name, SKU, category, and slug search.
- Synchronizes a changed catalog item to an exact matching database product/add-on where available.

SAFETY
------
- Creates timestamped protected catalog backups before catalog writes.
- Uses atomic file replacement and retains the latest 25 catalog backups.
- Uses prepared statements and strict table/column allowlists.
- Validates nonnegative prices up to $1,000,000.
- Adds CSRF protection and an operator confirmation prompt.
- Writes success/failure records to a JSONL audit log.
- Leaves completed orders, payments, subscriptions, and monitoring accounts unchanged.

CUMULATIVE BASELINE
-------------------
- Includes the idempotent KLEVR 2.0.0 10 percent equipment-price reduction migration.
- Minimum supported installed version remains 1.9.9.
- The 2.0.0 marker prevents the equipment reduction from running twice.

VALIDATION
----------
- PHP syntax validation passed for every packaged PHP file.
- Managed manifest checksums were generated for every payload, migration, and health file.
- Manifest self-checksum is intentionally excluded so checksum verification remains deterministic.
- ZIP archive integrity passed.
- No database credentials, customer records, uploads, payments, API keys, or protected media are bundled.

INSTALLATION
------------
1. Open Admin > Software Updates.
2. Upload {package_path.name}.
3. Confirm current version is at least 1.9.9 and target version is 2.1.0.
4. Choose Back Up & Apply.
5. Confirm all post-install health checks pass.
6. Open Admin > Pricing Manager.
7. Change one package or equipment price and save.
8. Confirm that price in Packages, System Builder, cart, and checkout.

A live production installation, checkout, or payment was not performed during packaging.
"""
    (DIST / "klevr-2.1.0-implementation-report.txt").write_text(report, encoding="utf-8")


def main() -> None:
    validate_php()
    manifest = build_manifest()
    package_path, digest = build_archive()
    write_report(package_path, digest, manifest)
    print(digest)


if __name__ == "__main__":
    main()
