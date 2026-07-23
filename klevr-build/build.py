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
PAYLOAD = SOURCE / "payload"
DIST = ROOT / "dist"
VERSION = "2.1.1"
PACKAGE_ID = "klevr-central-pricing-manager"
PACKAGE_NAME = f"klevr-software-update-{VERSION}-pricing-manager.zip"
MIGRATIONS = [
    {
        "id": "20260723_003_equipment_price_reduction",
        "up": "migrations/20260723_003_equipment_price_reduction.php",
    },
    {
        "id": "20260723_004_pricing_manager",
        "up": "migrations/20260723_004_pricing_manager.php",
    },
]
HEALTH_CHECKS = ["health/release-2.1.1.php"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_php() -> None:
    for path in sorted(SOURCE.rglob("*.php")):
        subprocess.run(["php", "-l", str(path)], check=True, capture_output=True, text=True)


def build_checksums() -> dict[str, str]:
    checksums: dict[str, str] = {}

    # The managed updater indexes payload files by their installed path, not by
    # their ZIP path. Therefore payload checksum keys must omit "payload/".
    for path in sorted(PAYLOAD.rglob("*")):
        if path.is_file():
            checksums[path.relative_to(PAYLOAD).as_posix()] = sha256(path)

    # Migration and health paths remain package-root relative.
    for migration in MIGRATIONS:
        up_path = SOURCE / migration["up"]
        checksums[migration["up"]] = sha256(up_path)
    for health_path in HEALTH_CHECKS:
        checksums[health_path] = sha256(SOURCE / health_path)

    return checksums


def build_manifest() -> dict:
    manifest = {
        "schema_version": 1,
        "package_id": PACKAGE_ID,
        "version": VERSION,
        "minimum_version": "1.9.9",
        "description": "Add the central administrator Pricing Manager and correct the managed updater manifest schema.",
        "payload_path": "payload",
        "requirements": {
            "php": "7.4.0",
            "extensions": ["pdo", "zip", "openssl"],
        },
        "delete_paths": [],
        "migrations": MIGRATIONS,
        "health_checks": HEALTH_CHECKS,
        "checksums": build_checksums(),
        "built_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    (SOURCE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def validate_preflight_schema(manifest: dict) -> None:
    errors: list[str] = []
    checksums = manifest.get("checksums", {})

    for path in sorted(PAYLOAD.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(PAYLOAD).as_posix()
        expected = checksums.get(relative_path)
        if not expected:
            errors.append(f"Managed package checksum is missing for {relative_path}")
        elif expected.lower() != sha256(path).lower():
            errors.append(f"Managed package checksum does not match for {relative_path}")

    for migration in manifest.get("migrations", []):
        if not isinstance(migration, dict) or not migration.get("id") or not migration.get("up"):
            errors.append("Every migration requires an id and up file.")
            continue
        up_path = SOURCE / migration["up"]
        if not up_path.is_file():
            errors.append(f"Migration up file is missing for {migration['id']}")
        expected = checksums.get(migration["up"])
        if not expected:
            errors.append(f"Managed package checksum is missing for {migration['up']}")
        elif up_path.is_file() and expected.lower() != sha256(up_path).lower():
            errors.append(f"Managed package checksum does not match for {migration['up']}")

    for health_path in manifest.get("health_checks", []):
        path = SOURCE / health_path
        expected = checksums.get(health_path)
        if not path.is_file():
            errors.append(f"Health check is missing: {health_path}")
        elif not expected:
            errors.append(f"Managed package checksum is missing for {health_path}")
        elif expected.lower() != sha256(path).lower():
            errors.append(f"Managed package checksum does not match for {health_path}")

    if errors:
        raise RuntimeError("Preflight schema validation failed:\n" + "\n".join(errors))


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
    payload_files = [path for path in PAYLOAD.rglob("*") if path.is_file()]
    report = f"""KLEVR 2.1.1 — PRICING MANAGER PREFLIGHT FIX
================================================

Package: {package_path.name}
Minimum version: {manifest['minimum_version']}
Target version: {manifest['version']}
Payload files: {len(payload_files)}
Database migrations: {len(manifest['migrations'])}
Deleted files: {len(manifest['delete_paths'])}
SHA-256: {digest}

CORRECTED
---------
- Payload checksum keys are relative to the payload root and omit the payload/ prefix.
- Every migration is declared with the required id and up fields.
- Migration and health-check checksums remain package-root relative.
- The package remains cumulative from KLEVR 1.9.9 and includes the Pricing Manager.

VALIDATION
----------
- Every packaged PHP file passed php -l.
- Payload, migration, and health-check SHA-256 validation passed.
- Managed updater preflight schema emulation passed.
- ZIP archive integrity passed.

Discard the rejected 2.1.0 stage and upload this 2.1.1 package through Admin > Software Updates.
A live production installation, checkout, payment, or database migration was not performed during packaging.
"""
    (DIST / "klevr-2.1.1-preflight-fix-report.txt").write_text(report, encoding="utf-8")


def main() -> None:
    validate_php()
    manifest = build_manifest()
    validate_preflight_schema(manifest)
    package_path, digest = build_archive()
    write_report(package_path, digest, manifest)
    print(digest)


if __name__ == "__main__":
    main()
