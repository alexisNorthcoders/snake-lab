"""Rewards, computed from the engine's step facts (the engine never computes them).

A personality is a function from one step's facts (`infos` of `SnakeVectorEnv.step`: an int32 array a field, one
entry a match) to a float32 reward a match. `REWARDS` names them; a new personality is a new function there.
"""

from collections.abc import Callable, Mapping

import numpy as np

Facts = Mapping[str, np.ndarray]


def glutton_reward(
    facts: Facts,
    food_weight: float = 1.0,
    tick_bonus: float = 0.1,
    win_bonus: float = 0.0,
    draw_penalty: float = 0.0,
    loss_penalty: float = 0.0,
) -> np.ndarray:
    """Food score gained this tick, a small bonus for still being alive after it, and on the tick the round ends a win
    bonus, or a penalty for a draw or a loss.

    The first three terms are the neuroevolution Glutton's fitness (food score, ticks survived, wins), a tick at a time.
    The penalties make a draw cost something; keep the loss penalty at least the draw one, or a snake heading for a
    draw learns to die instead.
    """
    ended = facts["ended"] == 1
    won, lost, drawn = ended & (facts["outcome"] == 1), ended & (facts["outcome"] == 2), ended & (facts["outcome"] == 3)
    return (
        food_weight * facts["scoreGained"] + tick_bonus * facts["alive"]
        + win_bonus * won - draw_penalty * drawn - loss_penalty * lost
    ).astype(np.float32)


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


# Each takes the facts and the run's settings (a dict), and picks its own weights from it. `alone` (true in the alone
# stage) drops the Glutton's draw and loss penalties: alone, every round ends drawn or lost, with no one to beat.
REWARDS: dict[str, Callable[[Facts, Mapping], np.ndarray]] = {
    "glutton": lambda facts, s: glutton_reward(
        facts, s["food_weight"], s["tick_bonus"], s.get("glutton_win_bonus", 0.0),
        0.0 if s.get("alone") else s.get("glutton_draw_penalty", 0.0),
        0.0 if s.get("alone") else s.get("glutton_loss_penalty", 0.0),
    ),
    "survivor": lambda facts, s: survivor_reward(facts, s["alive_weight"], s["survivor_win_bonus"]),
    "hunter": lambda facts, s: hunter_reward(facts, s["kill_bonus"], s["hunter_win_bonus"], s["food_bonus"]),
}
