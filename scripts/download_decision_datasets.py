"""Download pinned Choice, Noul and Score datasets without training packages."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


BANKING_REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
GO_EMOTIONS_REVISION = "add492243ff905527e67aeb8b80c082af02207c3"
SST5_REVISION = "e51bdcd8cd3a30da231967c1a249ba59361279a3"
SST5_LEVELS = ["very negative", "negative", "neutral", "positive", "very positive"]


def download(url: str, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve an existing download; its digest remains recorded in the manifest.
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".part")
        with urlopen(url, timeout=120) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(path)
    if path.stat().st_size == 0:
        raise ValueError(f"Empty dataset file: {path}")
    return {
        "path": path.name,
        "url": url,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".cache/jevlike-training/datasets"))
    args = parser.parse_args()
    banking_base = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{BANKING_REVISION}"
    emotions_base = f"https://huggingface.co/datasets/google-research-datasets/go_emotions/resolve/{GO_EMOTIONS_REVISION}"
    sst5_base = f"https://huggingface.co/datasets/SetFit/sst5/resolve/{SST5_REVISION}"
    specs = [
        ("banking77", BANKING_REVISION, "https://github.com/PolyAI-LDN/task-specific-datasets", "CC-BY-4.0", {
            "train.csv": f"{banking_base}/banking_data/train.csv",
            "test.csv": f"{banking_base}/banking_data/test.csv",
            "categories.json": f"{banking_base}/banking_data/categories.json",
            "LICENSE": f"{banking_base}/LICENSE",
        }),
        ("go_emotions", GO_EMOTIONS_REVISION, "https://huggingface.co/datasets/google-research-datasets/go_emotions", "Apache-2.0 (dataset card)", {
            "train.parquet": f"{emotions_base}/simplified/train-00000-of-00001.parquet",
            "validation.parquet": f"{emotions_base}/simplified/validation-00000-of-00001.parquet",
            "test.parquet": f"{emotions_base}/simplified/test-00000-of-00001.parquet",
            "README.md": f"{emotions_base}/README.md",
        }),
        ("sst5", SST5_REVISION, "https://huggingface.co/datasets/SetFit/sst5", "Not specified in SetFit/sst5 repository", {
            "train.jsonl": f"{sst5_base}/train.jsonl",
            "validation.jsonl": f"{sst5_base}/dev.jsonl",
            "test.jsonl": f"{sst5_base}/test.jsonl",
            "README.md": f"{sst5_base}/README.md",
        }),
    ]
    for name, revision, source, license_name, files in specs:
        directory = args.output / name
        manifest = {"dataset": name, "config": "simplified" if name == "go_emotions" else "default",
                    "revision": revision, "source": source, "license": license_name, "files": []}
        for filename, url in files.items():
            manifest["files"].append(download(url, directory / filename))
        if name == "sst5":
            manifest["primitive"] = "score"
            manifest["levels"] = SST5_LEVELS
            manifest["paper"] = "https://aclanthology.org/D13-1170/"
            manifest["rows"] = {}
            for split in ("train", "validation", "test"):
                rows = [json.loads(line) for line in (directory / f"{split}.jsonl").read_text().splitlines() if line.strip()]
                for row in rows:
                    label = row["label"]
                    if not isinstance(label, int) or not 0 <= label < len(SST5_LEVELS):
                        raise ValueError(f"Invalid SST-5 label in {split}: {label!r}")
                    if row["label_text"] != SST5_LEVELS[label] or not isinstance(row["text"], str):
                        raise ValueError(f"Unexpected SST-5 schema in {split}")
                manifest["rows"][split] = len(rows)
        (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        print(f"{name}: {len(files)} files -> {directory.resolve()}")


if __name__ == "__main__":
    main()
