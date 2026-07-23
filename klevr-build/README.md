# KLEVR Managed Update Builder

This temporary build branch produces the cumulative KLEVR 2.1.0 Pricing Manager update.

## Build

```bash
python klevr-build/build.py
```

The builder validates all packaged PHP files, writes a managed-update manifest, creates the ZIP archive, verifies ZIP integrity, and writes the SHA-256 file and implementation report to `klevr-build/dist/`.
