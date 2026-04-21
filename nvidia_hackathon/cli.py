from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import prepare_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAFE_ROOT = Path("/gpfs/public/datasets/Video-MME/processed_data/short/video")
DEFAULT_SAFEWATCH_ROOT = Path("/gpfs/public/datasets/SafeWatch-Bench")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "inserted_c123_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a synthetic eval manifest: 90 videos with 0/1/2 C1–C3 harmful inserts "
            "(legacy schedule for dataset construction)."
        )
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest-json", type=Path, default=None)
    parser.add_argument("--schedule-summary-json", type=Path, default=None)

    parser.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    parser.add_argument("--safewatch-root", type=Path, default=DEFAULT_SAFEWATCH_ROOT)

    parser.add_argument("--num-zero-insert", type=int, default=30)
    parser.add_argument("--num-single-per-category", type=int, default=10)
    parser.add_argument("--num-pair-per-combo", type=int, default=10)
    parser.add_argument("--base-safe-clip-duration", type=float, default=30.0)
    parser.add_argument("--min-safe-clip-duration", type=float, default=30.0)
    parser.add_argument("--max-insert-duration", type=float, default=None)

    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=128)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest_json or (args.output_root / "dataset_manifest.json")
    schedule_summary_path = args.schedule_summary_json or (args.output_root / "schedule_summary.json")

    samples = prepare_dataset(
        output_root=args.output_root,
        manifest_path=manifest_path,
        schedule_summary_path=schedule_summary_path,
        safe_root=args.safe_root,
        safewatch_root=args.safewatch_root,
        num_zero_insert=args.num_zero_insert,
        num_single_per_category=args.num_single_per_category,
        num_pair_per_combo=args.num_pair_per_combo,
        base_safe_clip_duration_sec=args.base_safe_clip_duration,
        min_safe_clip_duration=args.min_safe_clip_duration,
        max_insert_duration=args.max_insert_duration,
        fps=args.fps,
        max_frames=args.max_frames,
        seed=args.seed,
    )
    print(f"Prepared {len(samples)} samples at {manifest_path}")
    print(f"Saved schedule summary to {schedule_summary_path}")


if __name__ == "__main__":
    main()
