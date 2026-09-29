"""Rewards, computed from the engine's step facts (the engine never computes them).

A personality is a function from one step's facts (`infos` of `SnakeVectorEnv.step`: an int32 array a field, one
entry a match) to a float32 reward a match. `REWARDS` names them; a new personality is a new function there.
"""

from collections.abc import Callable, Mapping

import numpy as np

Facts = Mapping[str, np.ndarray]


def glutton_reward(facts: Facts, food_weight: float = 1.0, tick_bonus: float = 0.1) -> np.ndarray:
    """Food score gained this tick, plus a small bonus for still being alive after it.

    The same two terms as the neuroevolution Glutton's fitness (food score, ticks survived), a tick at a time.
    """
    return (food_weight * facts["scoreGained"] + tick_bonus * facts["alive"]).astype(np.float32)


def survivor_reward(facts: Facts, alive_weight: float = 0.1, win_bonus: float = 10.0) -> np.ndarray:
    """A bonus for each tick still alive after it, and a win bonus on the tick the round ends with the learner winning."""
    won = (facts["ended"] == 1) & (facts["outcome"] == 1)
    return (alive_weight * facts["alive"] + win_bonus * won).astype(np.float32)


def hunter_reward(facts: Facts, kill_bonus: float = 10.0, win_bonus: float = 5.0, food_bonus: float = 0.01) -> np.ndarray:
    """A big bonus a kill, a win bonus at the end, and a small one for food score gained.

    `kills` is the engine's count by the lab's rule: a head-on collision kills for neither snake, so it's no kill here.
    """
    won = (facts["ended"] == 1) & (facts["outcome"] == 1)
    return (kill_bonus * facts["kills"] + win_bonus * won + food_bonus * facts["scoreGained"]).astype(np.float32)


# Each takes the facts and the run's settings (a dict), and picks its own weights from it.
REWARDS: dict[str, Callable[[Facts, Mapping], np.ndarray]] = {
    "glutton": lambda facts, s: glutton_reward(facts, s["food_weight"], s["tick_bonus"]),
    "survivor": lambda facts, s: survivor_reward(facts, s["alive_weight"], s["survivor_win_bonus"]),
    "hunter": lambda facts, s: hunter_reward(facts, s["kill_bonus"], s["hunter_win_bonus"], s["food_bonus"]),
}
