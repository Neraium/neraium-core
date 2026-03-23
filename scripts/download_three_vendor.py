#!/usr/bin/env python3
"""Download three@0.162.0 from the npm registry into apps/api/static/vendor/three.

No npm required — uses the official .tgz (build/ + examples/jsm for addons).
Run from repo root: python scripts/download_three_vendor.py
"""
from __future__ import annotations

import io
import sys
import tarfile
import urllib.request
from pathlib import Path

THREE_VERSION = "0.162.0"
URL = f"https://registry.npmjs.org/three/-/three-{THREE_VERSION}.tgz"

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "apps" / "api" / "static" / "vendor" / "three"


def main() -> int:
    prefix = "package/"
    wanted_prefixes = ("build/", "examples/")

    print(f"Downloading three@{THREE_VERSION} …")
    req = urllib.request.Request(URL, headers={"User-Agent": "neraium-core-vendor-three"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = resp.read()

    print(f"Extracting to {TARGET} …")
    if TARGET.exists():
        import shutil

        shutil.rmtree(TARGET)

    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            name = m.name
            if not name.startswith(prefix):
                continue
            rel = name[len(prefix) :]
            if not rel.startswith(wanted_prefixes):
                continue
            dest = TARGET / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(m)
            if src:
                dest.write_bytes(src.read())

    if not (TARGET / "build" / "three.module.js").is_file():
        print("error: three.module.js missing after extract", file=sys.stderr)
        return 1

    print("ok:", TARGET / "build" / "three.module.js")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
