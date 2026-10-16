"""
Sensor Fusion Lab -- command-line entry point.

Runs the full IMU + GPS Kalman-filter demo headlessly (no display needed).

    python main.py            # full 120 s scenario, plots -> results/
    python main.py --quick     # 30 s scenario, same story, faster
    python main.py --seed 7     # reproducible run with a different RNG seed
"""

import argparse

from demo import main as run_demo


def parse_args():
    p = argparse.ArgumentParser(description="Sensor Fusion Lab demo")
    p.add_argument("--quick", action="store_true",
                   help="run a shorter (30 s) version of the demo")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for the sensor noise (default: 0)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        run_demo(duration=30.0, seed=args.seed, outages=((12.0, 20.0),))
    else:
        run_demo(duration=120.0, seed=args.seed,
                 outages=((40.0, 50.0), (85.0, 95.0)))


if __name__ == "__main__":
    main()
