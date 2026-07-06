"""Pre-fetch pinned model weights into ./weights (the ONLY place network is
allowed, and only at setup time — never during pipeline runs).

  uv run python tools/fetch_weights.py            # verify/fetch per pins.json
  uv run python tools/fetch_weights.py --write-pins  # first-time pin (records
      resolved HF revision + sha256 + card license; refuses non-allowed licenses)
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scenic import hashing, schema  # noqa: E402

WEIGHTS_DIR = REPO_ROOT / "weights"
ALLOWED = {"apache-2.0": "Apache-2.0", "mit": "MIT", "bsd-3-clause": "BSD-3-Clause"}

WANTED = {
    "depth_anything_v2_small": {
        "repo": "depth-anything/Depth-Anything-V2-Small-hf",
        "files": ["config.json", "model.safetensors", "preprocessor_config.json"],
    },
    "rtdetr_r18": {
        "repo": "PekingU/rtdetr_r18vd",
        "files": ["config.json", "model.safetensors", "preprocessor_config.json"],
    },
}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "scenic-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _api(repo: str) -> dict:
    return json.loads(_get(f"https://huggingface.co/api/models/{repo}"))


def _download(repo: str, rev: str, rel: str, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{repo}/resolve/{rev}/{rel}"
    print(f"  fetch {url}")
    data = _get(url)
    dest.write_bytes(data)
    return hashing.sha256_bytes(data)


def write_pins() -> None:
    pins = {}
    for key, spec in WANTED.items():
        info = _api(spec["repo"])
        card_license = (info.get("cardData") or {}).get("license", "")
        if card_license not in ALLOWED:
            raise SystemExit(
                f"REFUSING {spec['repo']}: card license {card_license!r} not in {sorted(ALLOWED)}"
            )
        rev = info["sha"]
        files = {}
        for rel in spec["files"]:
            files[rel] = _download(spec["repo"], rev, rel, WEIGHTS_DIR / key / rel)
        pins[key] = {
            "repo": spec["repo"],
            "revision": rev,
            "license": ALLOWED[card_license],
            "license_url": f"https://huggingface.co/{spec['repo']}",
            "files": files,
        }
        print(f"  pinned {key} @ {rev} ({ALLOWED[card_license]})")
    schema.write_validated(WEIGHTS_DIR / "pins.json", pins, "pins")
    print(f"wrote {WEIGHTS_DIR / 'pins.json'}")


def fetch_from_pins() -> None:
    pins = schema.read_validated(WEIGHTS_DIR / "pins.json", "pins")
    for key, pin in pins.items():
        for rel, want in sorted(pin["files"].items()):
            dest = WEIGHTS_DIR / key / rel
            if dest.exists() and hashing.sha256_file(dest) == want:
                print(f"  ok {key}/{rel}")
                continue
            got = _download(pin["repo"], pin["revision"], rel, dest)
            if got != want:
                raise SystemExit(f"hash mismatch {key}/{rel}: {got} != {want}")
            print(f"  ok {key}/{rel} (fetched)")
    print("weights verified against pins.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-pins", action="store_true")
    args = ap.parse_args()
    write_pins() if args.write_pins else fetch_from_pins()
