#!/usr/bin/env python3
"""
run_trial.py — Execute all module trial/verification blocks sequentially.

Runs each module's `if __name__ == '__main__':` trial block in dependency order.
Reports aggregate PASS/FAIL results.

Usage:
    python scripts/run_trial.py
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import List, Tuple

# Module execution order (dependency-ordered)
MODULE_ORDER: List[Tuple[str, str]] = [
    ("config", "config.py"),
    ("utils", "src/microex/utils.py"),
    ("capture", "src/microex/capture.py"),
    ("face_detector", "src/microex/face_detector.py"),
    ("landmarks", "src/microex/landmarks.py"),
    ("motion_features", "src/microex/motion_features.py"),
    ("apex_spotter", "src/microex/apex_spotter.py"),
    ("classifier", "src/microex/classifier.py"),
    ("logger", "src/microex/logger.py"),
    ("pipeline", "src/microex/pipeline.py"),
]


def run_module_trial(
    module_name: str,
    module_path: str,
    project_root: Path,
) -> Tuple[bool, str]:
    """
    Run a single module's trial block.

    Args:
        module_name: Human-readable module name.
        module_path: Relative path to the module file.
        project_root: Project root directory.

    Returns:
        Tuple of (passed: bool, output: str).
    """
    full_path = project_root / module_path
    if not full_path.exists():
        return False, f"Module file not found: {full_path}"

    try:
        result = subprocess.run(
            [sys.executable, str(full_path)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(project_root),
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )

        output = result.stdout + result.stderr

        # Check exit code
        if result.returncode != 0:
            return False, output

        # Check for FAIL in output
        if "FAIL:" in output:
            return False, output

        return True, output

    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT: {module_name} trial exceeded 120s"
    except Exception as e:
        return False, f"ERROR: {e}"


def main() -> int:
    """Run all module trials and report results."""
    project_root = Path(__file__).resolve().parent.parent
    
    print("=" * 70)
    print("  MICRO-EXPRESSION DETECTION SYSTEM — MODULE TRIAL RUNNER")
    print("=" * 70)
    print(f"  Project root: {project_root}")
    print(f"  Python: {sys.executable}")
    print(f"  Modules to test: {len(MODULE_ORDER)}")
    print("=" * 70)
    print()

    results: List[Tuple[str, bool, str]] = []

    for module_name, module_path in MODULE_ORDER:
        print(f"{'─' * 60}")
        print(f"  Running trial: {module_name} ({module_path})")
        print(f"{'─' * 60}")

        passed, output = run_module_trial(module_name, module_path, project_root)
        results.append((module_name, passed, output))

        # Print trial output (indented)
        for line in output.strip().split("\n"):
            print(f"    {line}")

        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"\n  [{status}] {module_name}")
        print()

    # ── Summary ────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  TRIAL RESULTS SUMMARY")
    print("=" * 70)

    passed_count = sum(1 for _, p, _ in results if p)
    total = len(results)

    for module_name, passed, _ in results:
        icon = "✓" if passed else "✗"
        status = "PASS" if passed else "FAIL"
        print(f"  {icon} {module_name:20s} [{status}]")

    print(f"\n  Total: {passed_count}/{total} modules passed")

    if passed_count == total:
        print("\n  ★ ALL MODULES PASSED — Ready for pipeline integration")
    else:
        failed = [name for name, p, _ in results if not p]
        print(f"\n  ⚠ FAILED MODULES: {', '.join(failed)}")
        print("  Fix failing modules before proceeding to pipeline integration.")

    print("=" * 70)
    return 0 if passed_count == total else 1


if __name__ == "__main__":
    sys.exit(main())
