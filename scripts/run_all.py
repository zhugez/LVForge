#!/usr/bin/env python
"""Run ALL experiment variants in one command.

Usage:
    python scripts/run_all.py              # run all 5 experiments
    python scripts/run_all.py --only arcface triplet   # run selected ones
    python scripts/run_all.py --backup     # run all + backup weights to Google Drive
"""

import argparse
import subprocess
import sys
import time
import os

ALL_LOSSES = ["baseline", "arcface", "contrastive", "triplet", "multi_similarity"]


def run_experiment(loss_name: str) -> bool:
    """Run a single experiment variant. Returns True on success."""
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {loss_name}")
    print(f"{'='*60}\n")

    start = time.time()
    result = subprocess.run(
        [sys.executable, "scripts/train_flax.py", "--loss", loss_name],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    elapsed = time.time() - start
    status = "OK" if result.returncode == 0 else "FAIL"
    print(f"\n  [{status}] {loss_name} finished in {elapsed:.1f}s (exit={result.returncode})")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run all experiments sequentially.")
    parser.add_argument("--only", nargs="+", choices=ALL_LOSSES,
                        help="Run only these loss variants.")
    parser.add_argument("--backup", action="store_true",
                        help="Run backup_full.py after all experiments.")
    parser.add_argument("--gdrive", action="store_true",
                        help="Upload backup to Google Drive (requires --account).")
    parser.add_argument("--account", type=str, default=None,
                        help="Google account for Drive upload.")
    parser.add_argument("--folder-id", type=str, default=None,
                        help="Google Drive folder ID.")
    args = parser.parse_args()

    losses = args.only or ALL_LOSSES
    results = {}

    print(f"Running {len(losses)} experiment(s): {', '.join(losses)}\n")
    total_start = time.time()

    for loss in losses:
        results[loss] = run_experiment(loss)

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY  ({total_elapsed:.1f}s total)")
    print(f"{'='*60}")
    for loss, ok in results.items():
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {loss}")

    failed = [k for k, v in results.items() if not v]
    if failed:
        print(f"\n  {len(failed)} experiment(s) failed: {', '.join(failed)}")

    # Optional backup
    if args.backup or args.gdrive:
        print(f"\n{'='*60}")
        print(f"  BACKUP")
        print(f"{'='*60}\n")
        backup_cmd = [sys.executable, "backup_full.py"]
        if args.gdrive:
            backup_cmd.append("--gdrive")
            if args.account:
                backup_cmd.extend(["--account", args.account])
            if args.folder_id:
                backup_cmd.extend(["--folder-id", args.folder_id])
        subprocess.run(
            backup_cmd,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
