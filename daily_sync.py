#!/usr/bin/env python3
"""
Daily incremental sync for alpha scores.

Default behavior:
- Sync only today
- Skip automatically when target date has no market data
"""

import argparse
import sys
import time

from score_all_dates import ensure_latest_available_score


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run daily incremental score sync into DuckDB."
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target date in YYYY-MM-DD format. Default: today.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    try:
        ensure_latest_available_score(args.date)
        print(f"Daily sync finished in {time.time() - t0:.2f}s")
        return 0
    except Exception as exc:
        print(f"Daily sync failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
