#!/usr/bin/env python3
"""
ExperimentAdder.py
Creates a new Experiments/ExpN folder by copying Exp001 as a starting point.

Usage:
    python ExperimentAdder.py           # auto-detects next experiment number
    python ExperimentAdder.py 3         # creates Exp003
    python ExperimentAdder.py --dry-run # preview without changes
"""

import os
import re
import sys
import shutil

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
EXP_DIR      = os.path.join(SCRIPT_DIR, "Experiments")
TEMPLATE_EXP = os.path.join(EXP_DIR, "Exp001")


def get_next_experiment() -> int:
    if not os.path.exists(EXP_DIR):
        return 2
    existing = [
        d for d in os.listdir(EXP_DIR)
        if re.match(r"Exp\d+$", d) and os.path.isdir(os.path.join(EXP_DIR, d))
    ]
    if not existing:
        return 2
    numbers = [int(re.search(r"\d+", d).group()) for d in existing]
    return max(numbers) + 1


def update_config_name(dst_dir: str, exp_name: str, dry_run: bool) -> None:
    config_path = os.path.join(dst_dir, "config.yaml")
    if not os.path.exists(config_path):
        return
    with open(config_path, encoding="utf-8") as f:
        content = f.read()
    updated = re.sub(r'(name:\s*")[^"]*(")', rf'\g<1>{exp_name}\2', content)
    if updated != content:
        print(f"  [UPDATE] experiment.name → {exp_name}")
        if not dry_run:
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(updated)


def main():
    dry_run = "--dry-run" in sys.argv
    args    = [a for a in sys.argv[1:] if not a.startswith("--")]

    if args:
        try:
            number = int(args[0])
            if number < 2:
                raise ValueError
        except ValueError:
            print(f"Error: '{args[0]}' is not a valid experiment number (must be ≥ 2).")
            sys.exit(1)
    else:
        number = get_next_experiment()

    exp_name = f"Exp{number:03d}"
    dst_dir  = os.path.join(EXP_DIR, exp_name)

    print(f"{'[DRY-RUN] ' if dry_run else ''}Creating: {exp_name}")

    if not os.path.exists(TEMPLATE_EXP):
        print(f"Error: Exp001 not found at {TEMPLATE_EXP}")
        sys.exit(1)

    if os.path.exists(dst_dir):
        print(f"Error: {exp_name} already exists: {dst_dir}")
        sys.exit(1)

    print(f"  [COPY] Exp001 → {exp_name}")
    if not dry_run:
        shutil.copytree(TEMPLATE_EXP, dst_dir)

    update_config_name(dst_dir, exp_name, dry_run)

    print(f"\n{'[DRY-RUN] ' if dry_run else ''}Done: {dst_dir}")
    print(f"Edit config, then run:")
    print(f"  python NeuralNetworks/train.py --config {os.path.relpath(dst_dir)}/config.yaml")


if __name__ == "__main__":
    main()
