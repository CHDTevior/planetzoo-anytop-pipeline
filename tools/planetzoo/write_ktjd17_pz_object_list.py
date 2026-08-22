"""Write the 311 Planet Zoo source object names specified by a KTJD-17 joint table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-spec", required=True, type=Path)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = json.loads(args.joint_spec.read_text(encoding="utf-8"))
    rigs = sorted(name for name in spec["rigs"] if name.startswith("PZ_"))
    objects = [f"{rig[3:]}.ovl" for rig in rigs]
    missing = [name for name in objects if not (args.input_root / name).is_dir()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} protocol rigs missing from source, e.g. {missing[:5]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(objects) + "\n", encoding="utf-8")
    print(json.dumps({"pzrigrigs": len(rigs), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
