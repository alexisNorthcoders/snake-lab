"""Steps random actions and prints environment steps a second: `python -m snake_lab.bench`."""

import argparse
import time

import numpy as np

from .env import SnakeVectorEnv


def measure(batch: int, seconds: float, args: argparse.Namespace) -> float:
    env = SnakeVectorEnv(
        batch,
        encoder=args.encoder,
        opponents=[o for o in args.opponents.split(",") if o],
        seat="random" if args.opponents else 0,
    )
    try:
        rng = np.random.default_rng(1)
        env.reset(seed=1)
        for _ in range(5):
            env.step(rng.integers(0, 3, batch))
        steps, start = 0, time.perf_counter()
        while time.perf_counter() - start < seconds:
            env.step(rng.integers(0, 3, batch))
            steps += batch
        return steps / (time.perf_counter() - start)
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", default="1,4,16,64,256", help="comma-separated batch sizes")
    parser.add_argument("--seconds", type=float, default=5, help="seconds measured a batch size")
    parser.add_argument("--encoder", type=int, default=2, choices=(1, 2))
    parser.add_argument("--opponents", default="rookie", help="comma-separated roster ids or brain files; empty is alone")
    args = parser.parse_args()
    print(f"encoder v{args.encoder}, opponents: {args.opponents or 'none'}")
    print(f"{'batch':>6} {'steps/s':>10}")
    for batch in (int(b) for b in args.batches.split(",")):
        print(f"{batch:>6} {measure(batch, args.seconds, args):>10.0f}", flush=True)


if __name__ == "__main__":
    main()
