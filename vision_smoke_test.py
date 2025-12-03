#!/usr/bin/env python3
"""
Quick smoke test for the multimodal vision pipeline.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

from api.vision import VisionAnalyzer


def run_smoke_test(image_folder: pathlib.Path, limit: int | None = None) -> None:
    analyzer = VisionAnalyzer()
    image_paths = sorted(image_folder.glob("*.jpg"))
    if limit:
        image_paths = image_paths[:limit]

    if not image_paths:
        print(f"No images found in {image_folder}")
        return

    print(f"Running vision smoke test on {len(image_paths)} screenshots...")
    start = time.perf_counter()
    summary = []

    for path in image_paths:
        with path.open("rb") as fp:
            detections, metadata = analyzer.analyze(fp.read())
            summary.append(
                {
                    "file": path.name,
                    "detections": len(detections),
                    "metadata": metadata,
                    "labels": [det["label"] for det in detections],
                }
            )
            print(f"[{path.name}] detections: {len(detections)} runtime: {metadata['runtime_ms']:.1f}ms")

    elapsed = time.perf_counter() - start
    print(f"Completed in {elapsed:.1f}s")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the multimodal vision smoke test.")
    parser.add_argument(
        "--images",
        type=pathlib.Path,
        default=pathlib.Path("test/images"),
        help="Directory containing screenshots (default: test/images)",
    )
    parser.add_argument("--limit", type=int, help="Limit the number of screenshots processed.")
    args = parser.parse_args()

    run_smoke_test(args.images, args.limit)

