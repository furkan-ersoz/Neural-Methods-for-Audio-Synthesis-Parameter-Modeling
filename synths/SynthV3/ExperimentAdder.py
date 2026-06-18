#!/usr/bin/env python3
"""
ExperimentAdder.py
Creates a new Experiments/ExpN folder by copying an existing experiment.

Usage:
    python ExperimentAdder.py                  # next number, base = Exp001
    python ExperimentAdder.py 3                # creates Exp003 from Exp001
    python ExperimentAdder.py 15 --base Exp009 # creates Exp015 from Exp009
    python ExperimentAdder.py --base Exp009    # next number, base = Exp009
    python ExperimentAdder.py --dry-run        # preview without changes
"""

import os
import re
import sys
import shutil

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
EXP_DIR        = os.path.join(SCRIPT_DIR, "Experiments")
DEFAULT_BASE   = "Exp001"


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
    updated = re.sub(r'(name:\s*")[^"]*(")', rf'\g<1>{exp_name}\2', content, count=1)
    if updated != content:
        print(f"  [UPDATE] experiment.name → {exp_name}")
        if not dry_run:
            with open(config_path, "w", encoding="utf-8") as f:
                f.write(updated)


def _get_opt(flag: str, default: str) -> str:
    """Read '--flag value' from sys.argv."""
    if flag in sys.argv:
        i = sys.argv.index(flag)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def main():
    dry_run   = "--dry-run" in sys.argv
    base_name = _get_opt("--base", DEFAULT_BASE)

    # Positional args = numbers only (skip flags and their values)
    skip = {"--base"}
    args = []
    it = iter(sys.argv[1:])
    for a in it:
        if a == "--base":
            next(it, None)  # consume the value
            continue
        if a.startswith("--"):
            continue
        args.append(a)

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

    exp_name     = f"Exp{number:03d}"
    dst_dir      = os.path.join(EXP_DIR, exp_name)
    template_exp = os.path.join(EXP_DIR, base_name)

    print(f"{'[DRY-RUN] ' if dry_run else ''}Creating: {exp_name} (base: {base_name})")

    if not os.path.exists(template_exp):
        print(f"Error: base experiment '{base_name}' not found at {template_exp}")
        sys.exit(1)

    if os.path.exists(dst_dir):
        print(f"Error: {exp_name} already exists: {dst_dir}")
        sys.exit(1)

    print(f"  [COPY] {base_name} → {exp_name}")
    if not dry_run:
        shutil.copytree(template_exp, dst_dir)

    update_config_name(dst_dir, exp_name, dry_run)

    print(f"\n{'[DRY-RUN] ' if dry_run else ''}Done: {dst_dir}")


if __name__ == "__main__":
    main()
