"""Verify selected file hashes and reject unexpected tracked publication paths."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "PUBLICATION.json").read_text(encoding="utf-8"))
    records = manifest["files"]
    for record in records:
        path = ROOT / record["path"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != record["sha256"]:
            raise SystemExit(f"Hash mismatch: {record['path']}")
    expected = {record["path"] for record in records} | {"PUBLICATION.json"}
    expected.update(manifest["generated_files"])
    for name in manifest["generated_files"]:
        if not (ROOT / name).is_file():
            raise SystemExit(f"Missing generated figure: {name}")
    tracked = set(subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines())
    if tracked != expected:
        raise SystemExit(f"Tracked selection differs: extra={sorted(tracked-expected)}, missing={sorted(expected-tracked)}")
    print(f"Verified {len(records)} selected file hashes and {len(tracked)} tracked paths.")


if __name__ == "__main__":
    main()
