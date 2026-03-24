"""Entry point — runs simulation + webapp."""
from __future__ import annotations
import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description="Social Town Simulation")
    parser.add_argument("--agents", type=int, default=10)
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to snapshot JSON to resume from")
    args = parser.parse_args()
    print(f"[social-town] agents={args.agents} days={args.days}")
    # Replaced in Stage 5

if __name__ == "__main__":
    main()
