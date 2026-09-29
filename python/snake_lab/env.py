"""A Gymnasium vector environment over the lab's `env` subprocess (docs/env-protocol.md)."""

import os
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from gymnasium.vector import VectorEnv

from .protocol import EnvError, EnvProcessError, decode_batch, encode_frame, read_frame

ENCODER_SIZES = {1: 23, 2: 328}
REASON_TIME_UP = 2
REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_command() -> list[str]:
    # node with tsx loaded, not the `tsx` wrapper: one process, so killing it really stops the engine
    if (REPO_ROOT / "node_modules" / "tsx").exists():
        return ["node", "--import", "tsx", "src/cli/env.ts"]
    return ["npx", "tsx", "src/cli/env.ts"]


def _range(value: int | Sequence[int]) -> int | list[int]:
    return value if isinstance(value, int) else [int(value[0]), int(value[1])]


def _mode(modes: str | Sequence[str]) -> str:
    chosen = {modes} if isinstance(modes, str) else set(modes)
    if chosen == {"timed", "endless"}:
        return "mixed"
    if len(chosen) == 1 and chosen <= {"timed", "endless", "mixed"}:
        return chosen.pop()
    raise ValueError(f"modes must be timed, endless or both, got {modes!r}")


class SnakeVectorEnv(VectorEnv):
    """`num_envs` matches of the snake game, stepped as one batch.

    Actions are 0 left, 1 straight, 2 right. Rewards are always 0: the engine's facts come back in
    `infos`, one int32 array of length `num_envs` a field (see `FACT_FIELDS`). A match that ends is
    flagged in `terminated` (or `truncated` when time ran out) and its facts are those of its last
    tick, but the observation is already the next match's first. How a match ended is in `infos` too:
    `outcome` (1 won, 2 lost, 3 drawn) and `reason` (1 last-standing, 2 time-up, 3 learner-died), both 0
    for a match still going; `ended` is 1 on its last tick.
    """

    metadata = {"autoreset_mode": "SameStep"}
    _proc: subprocess.Popen | None = None
    _stderr: Any = None

    def __init__(
        self,
        num_envs: int = 8,
        *,
        encoder: int = 2,
        opponents: Sequence[str] = (),
        modes: str | Sequence[str] = ("timed", "endless"),
        learner_delay: int | Sequence[int] = 2,
        opponent_delay: int | Sequence[int] = 2,
        seat: int | str = 0,
        fps: int = 8,
        end_on_death: bool = True,
        command: Sequence[str] | None = None,
        cwd: str | os.PathLike | None = None,
    ) -> None:
        if encoder not in ENCODER_SIZES:
            raise ValueError(f"encoder must be 1 or 2, got {encoder!r}")
        if num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        self.num_envs = num_envs
        self.encoder = encoder
        self.single_observation_space = spaces.Box(-np.inf, np.inf, (ENCODER_SIZES[encoder],), np.float32)
        self.single_action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, (num_envs, ENCODER_SIZES[encoder]), np.float32)
        self.action_space = spaces.MultiDiscrete([3] * num_envs)
        self._config = {
            "encoder": encoder,
            "fps": fps,
            "endOnDeath": end_on_death,
            "matches": [
                {
                    "opponents": list(opponents),
                    "mode": _mode(modes),
                    "learnerDelay": _range(learner_delay),
                    "opponentDelay": _range(opponent_delay),
                    "seat": seat,
                }
            ]
            * num_envs,
        }
        self._command = list(command) if command is not None else _default_command()
        self._cwd = str(cwd) if cwd is not None else str(REPO_ROOT)
        self._seed_rng = np.random.default_rng()
        self._start()

    # subprocess ---------------------------------------------------------------------------

    def _start(self) -> None:
        self._stderr = tempfile.TemporaryFile()
        try:
            self._proc = subprocess.Popen(
                self._command, cwd=self._cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
                start_new_session=True,  # its own session: a Ctrl-C at the terminal reaches the trainer, not the engine
            )
        except OSError as error:
            self._stderr.close()
            raise EnvProcessError(f"could not start {self._command!r}: {error}") from error

    def _stderr_tail(self) -> str:
        try:
            self._stderr.seek(0)
            return self._stderr.read().decode("utf-8", "replace")[-2000:].strip()
        except (OSError, ValueError):
            return ""

    def _request(self, header: dict, payload: bytes = b"") -> tuple[dict, bytes]:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise EnvProcessError("the environment is closed")
        try:
            proc.stdin.write(encode_frame(header, payload))
            proc.stdin.flush()
            response, body = read_frame(proc.stdout)
        except (BrokenPipeError, EOFError, ValueError, OSError) as error:
            code = _reap(proc)
            detail = self._stderr_tail()
            self.close()
            raise EnvProcessError(
                f"the env subprocess died (exit {code})" + (f": {detail}" if detail else "")
            ) from error
        if response.get("type") == "error":
            raise EnvError(response.get("message", "unknown error"))
        return response, body

    def _batch(self, header: dict, payload: bytes = b"") -> tuple[np.ndarray, dict[str, Any]]:
        response, body = self._request(header, payload)
        if response.get("type") != "batch":
            self.close()
            raise EnvProcessError(f"expected a batch, got {response.get('type')!r}")
        facts, obs, fields = decode_batch(response, body)
        if obs.shape != self.observation_space.shape:
            self.close()
            raise EnvProcessError(f"observations are {obs.shape}, expected {self.observation_space.shape}")
        return obs, {name: facts[:, k].copy() for k, name in enumerate(fields)}

    def set_opponents(self, opponents: Sequence[Sequence[str]]) -> None:
        """Each match's own opponents (roster ids or brain files), used from the next `reset`."""
        if len(opponents) != self.num_envs:
            raise ValueError(f"expected opponents for {self.num_envs} matches, got {len(opponents)}")
        self._config["matches"] = [{**spec, "opponents": list(each)} for spec, each in zip(self._config["matches"], opponents)]

    # Gymnasium ----------------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is None:
            seed = int(self._seed_rng.integers(0, 2**32))
        elif not 0 <= seed < 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        obs, infos = self._batch({"type": "reset", "seed": int(seed), **self._config})
        return obs, infos

    def step(self, actions):
        actions = np.asarray(actions)
        if actions.shape != (self.num_envs,):
            raise ValueError(f"expected {self.num_envs} actions, got shape {actions.shape}")
        if not np.all((actions >= 0) & (actions <= 2)) or not np.all(actions == np.floor(actions)):
            raise ValueError("actions must be 0, 1 or 2")
        obs, infos = self._batch({"type": "step"}, actions.astype(np.uint8).tobytes())
        ended = infos["ended"] == 1
        truncated = ended & (infos["reason"] == REASON_TIME_UP)
        terminated = ended & ~truncated
        return obs, np.zeros(self.num_envs, dtype=np.float32), terminated, truncated, infos

    def close_extras(self, **kwargs: Any) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin and proc.poll() is None:
                try:
                    proc.stdin.write(encode_frame({"type": "close"}))
                    proc.stdin.flush()
                except OSError:
                    pass
            for stream in (proc.stdin, proc.stdout):
                try:
                    if stream:
                        stream.close()
                except OSError:
                    pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            if self._stderr is not None:
                self._stderr.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _reap(proc: subprocess.Popen) -> int | None:
    try:
        return proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            return proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            return None
