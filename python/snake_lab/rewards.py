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


# Each takes the facts and the run's settings (a dict), and picks its own weights from it.
REWARDS: dict[str, Callable[[Facts, Mapping], np.ndarray]] = {
    "glutton": lambda facts, s: glutton_reward(facts, s["food_weight"], s["tick_bonus"]),
}
