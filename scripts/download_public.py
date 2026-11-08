# -*- coding: utf-8 -*-
"""Fetch the public datasets and verify them against the recorded checksums.

Nothing is repackaged: each file is downloaded from its cited location, the raw
bytes are checked against a recorded SHA-256, and only then written to data/.

The wide-range database ships inside the archived release of the gpvisc package
(Zenodo DOI 10.5281/zenodo.13843250, version 0.3.5). The archive is a zip, so
the file is extracted from it rather than from a repository checkout.

    python scripts/download_public.py            # download missing files
    python scripts/download_public.py --check    # verify what is on disk
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import DATA  # noqa: E402

MANIFEST = DATA / "manifest.json"

SOURCES = {
    "gpvisc/dataset+sciglass_lp.csv": {
        "urls": [
            "https://zenodo.org/api/records/15682732/files/charlesll/gpvisc-v0.3.5.zip/content",
            "https://raw.githubusercontent.com/charlesll/gpvisc/v0.3.5/gpvisc/data/dataset+sciglass_lp.csv",
        ],
        "member": "gpvisc/data/dataset+sciglass_lp.csv",   # path inside the zip
        "sha256": "5a5bbf6e9e5a9048015f5e7724e644d73c312e3358b767d672085757ed2e58f3",
        "source": "gpvisc 0.3.5, DOI 10.5281/zenodo.13843250",
    },
}


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract(blob: bytes, member: str | None) -> bytes | None:
    """Return the payload, pulling it out of a zip when the source is an archive."""
    if not zipfile.is_zipfile(io.BytesIO(blob)):
        return blob
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = z.namelist()
        hit = next((n for n in names if n.endswith(member) if member), None)
        if hit is None:
            print(f"    archive does not contain {member}; it holds {len(names)} entries")
            return None
        return z.read(hit)


def fetch(rel: str, spec: dict, check_only: bool) -> bool:
    dst = DATA / rel
    want = spec.get("sha256")
    if dst.exists():
        got = sha256_of(dst)
        if want and got != want:
            print(f"  MISMATCH {rel}\n    expected {want}\n    found    {got}")
            return False
        print(f"  ok       {rel}  ({dst.stat().st_size} bytes)")
        return True
    if check_only:
        print(f"  missing  {rel}")
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    for url in spec["urls"]:
        try:
            print(f"  fetching {rel}")
            print(f"    <- {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "dstar-repro/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                blob = r.read()
            payload = extract(blob, spec.get("member"))
            if payload is None:
                continue
            got = sha256_bytes(payload)
            if want and got != want:
                print(f"    checksum mismatch: {got}")
                continue
            dst.write_bytes(payload)
            if not want:
                spec["sha256"] = got
                print(f"    recorded sha256 {got}")
            print(f"    ok, {len(payload)} bytes, sha256 verified")
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"    failed: {exc}")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only, no network")
    args = ap.parse_args()

    print(f"data root: {DATA}")
    ok = True
    for rel, spec in SOURCES.items():
        ok &= fetch(rel, spec, args.check)

    man = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    man.update({k: {kk: vv for kk, vv in v.items() if kk != "urls"} for k, v in SOURCES.items()})
    MANIFEST.write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(f"\nmanifest: {MANIFEST}")
    print("ALL CHECKS PASSED" if ok else "FAILURES ABOVE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
