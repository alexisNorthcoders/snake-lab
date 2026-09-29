"""The league: who a PPO batch meets. Snapshots of the policy itself (its earlier checkpoints' brains) and the roster.

PPO has no population, so the league's opponents are frozen copies of the policy from earlier checkpoints, played by the
TypeScript side as brain files, and every roster snake (the rookie included). Which ones a batch meets is drawn from
the seed, so a resumed run meets the same ones.
"""

import json
import subprocess
from pathlib import Path

import numpy as np

from .env import REPO_ROOT


def roster_ids() -> list[str]:
    """The pinned `snake-colyseus` package's roster ids, sorted, asked of the TypeScript side."""
    command = ["node", "--import", "tsx", "src/cli/roster.ts"] if (REPO_ROOT / "node_modules" / "tsx").exists() else ["npx", "tsx", "src/cli/roster.ts"]
    done = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    if done.returncode != 0:
        raise RuntimeError(f"could not read the roster: {done.stderr.strip()}")
    return json.loads(done.stdout)


def snapshot_paths(generations: Path, before: int) -> list[str]:
    """Absolute paths of the brains of checkpoints 0 to `before` - 1: the policy's own past, frozen."""
    return [str((generations / f"gen-{g:04d}.json").resolve()) for g in range(before)]


def league_opponents(
    rng: np.random.Generator, matches: int, four_player_share: float, snapshots: list[str], roster: list[str]
) -> list[list[str]]:
    """The opponents of each of `matches` matches: `round(share * matches)` of them (drawn at random places) have three
    opponents, the rest one. Each opponent is a snapshot half the time (when there are any), else a roster snake.
    """
    if not roster:
        raise ValueError("the league needs the roster")
    four = set(rng.permutation(matches)[: round(four_player_share * matches)].tolist())
    chosen = []
    for match in range(matches):
        each = []
        for _ in range(3 if match in four else 1):
            from_snapshots = bool(snapshots) and rng.random() < 0.5
            pool = snapshots if from_snapshots else roster
            each.append(pool[int(rng.integers(len(pool)))])
        chosen.append(each)
    return chosen
