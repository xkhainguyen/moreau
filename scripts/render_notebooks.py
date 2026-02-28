#!/usr/bin/env python3
"""Execute all notebooks and extract thumbnail images for the README gallery."""

import subprocess
import sys
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"
ASSETS_DIR = Path(__file__).parent.parent / "assets"

NOTEBOOKS = [
    "portfolio_optimization.ipynb",
    "mpc_trajectory.ipynb",
    "compressed_sensing.ipynb",
    "optimal_transport.ipynb",
    "swarm_planning.ipynb",
    "predict_then_optimize.ipynb",
    "sudoku.ipynb",
    "contact_friction.ipynb",
    "bandwidth_allocation.ipynb",
]


def execute_notebook(nb_path: Path) -> bool:
    """Execute a notebook in place, returning True on success."""
    print(f"\n{'='*60}")
    print(f"Executing: {nb_path.name}")
    print(f"{'='*60}")

    result = subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=600",
            str(nb_path),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"  FAILED: {nb_path.name}")
        print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
        return False

    print(f"  OK: {nb_path.name}")
    return True


def main():
    ASSETS_DIR.mkdir(exist_ok=True)

    results = {}
    for nb_name in NOTEBOOKS:
        nb_path = NOTEBOOKS_DIR / nb_name
        if not nb_path.exists():
            print(f"  SKIP (not found): {nb_name}")
            results[nb_name] = None
            continue
        results[nb_name] = execute_notebook(nb_path)

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for nb_name, ok in results.items():
        status = "OK" if ok else ("SKIP" if ok is None else "FAIL")
        print(f"  [{status:4s}] {nb_name}")

    n_fail = sum(1 for v in results.values() if v is False)
    if n_fail:
        print(f"\n{n_fail} notebook(s) failed!")
        sys.exit(1)
    else:
        print("\nAll notebooks executed successfully!")

    # List generated assets
    print(f"\nAssets in {ASSETS_DIR}:")
    for f in sorted(ASSETS_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:40s} {size_kb:8.1f} KB")


if __name__ == "__main__":
    main()
