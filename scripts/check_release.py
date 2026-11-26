"""Verify the exact release allowlist and scan its text for obvious secrets."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "RELEASE_MANIFEST.json"
PATTERNS = [
    ("GitHub credential", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{40,}\b")),
    ("API credential", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("embedded URI credential", re.compile(r"(?:https?|bolt|neo4j)://[^\s/:@]+:[^\s/@]+@", re.I)),
    ("personal absolute path", re.compile(r"[A-Za-z]:[\\/](?:Users|HuaweiMoveData)[\\/]", re.I)),
]


def approved_files():
    manifest = json.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))
    entries = manifest["files"]
    names = [entry["path"] for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate manifest path")
    return manifest, entries


def verify():
    manifest, entries = approved_files()
    errors = []
    total_bytes = 0
    for entry in entries:
        name = entry["path"]
        relative = PurePosixPath(name)
        path = ROOT.joinpath(*relative.parts)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name:
            errors.append(f"Invalid manifest path: {name}")
            continue
        if not path.resolve().is_relative_to(ROOT.resolve()) or path.is_symlink():
            errors.append(f"Path leaves release directory: {name}")
            continue
        if not path.is_file():
            errors.append(f"Missing file: {name}")
            continue
        raw = path.read_bytes()
        total_bytes += len(raw)
        if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            errors.append(f"Fingerprint mismatch: {name}")
        if len(raw) > 1024 * 1024:
            errors.append(f"Unexpected large file: {name}")
        if path.suffix.lower() in {".docx", ".pdf", ".html", ".zip", ".bin", ".pt", ".pth", ".safetensors", ".sqlite", ".db"}:
            errors.append(f"Excluded file type: {name}")
        if any(part.lower() in {"private", "restricted", "raw", "models", "outputs", "__pycache__"} for part in relative.parts):
            errors.append(f"Excluded directory: {name}")
        if relative.parts[0] == "data" and relative.parts[:2] != ("data", "synthetic"):
            errors.append(f"Study data path is not permitted: {name}")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            errors.append(f"Unexpected binary content: {name}")
            continue
        for label, pattern in PATTERNS:
            if pattern.search(text):
                errors.append(f"Possible {label}: {name}")
    return {"passed": not errors, "manifest_version": manifest["version"],
            "allowlisted_files": len(entries) + 1, "bytes_excluding_manifest": total_bytes,
            "errors": errors,
            "scope": "Exact file/hash allowlist and obvious-secret checks; redistribution rights are documented separately"}


if __name__ == "__main__":
    report = verify()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    sys.exit(0 if report["passed"] else 1)
