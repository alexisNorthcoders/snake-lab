"""The PPO trainer: settings, the run folder (the neuroevolution layout) and the training loop.

    python -m snake_lab.run --personality glutton --run ../runs/ppo-1 --seed 1
    python -m snake_lab.run --resume ../runs/ppo-1

Training is a row of **checkpoints** (`checkpoint_every` PPO updates each), which the dashboard and `promote` see as
generations. Each is one stage: `alone` for the first `alone_updates`, then `rookie` (1v1 against the rookie) for
`rookie_updates`. At the start of each the env is reset from a seed drawn from (seed, checkpoint number) and so is the
sampling generator, so a run resumed from a checkpoint replays exactly the run that never stopped (on the same machine).
"""

import argparse
import json
import os
import signal
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch

from .brain import export_brain
from .env import ENCODER_SIZES, SnakeVectorEnv
from .ppo import Agent, collect, update
from .rewards import REWARDS


@dataclass
class Settings:
    personality: str = "glutton"
    seed: int = 1
    # stages, in PPO updates; each is a whole number of checkpoints
    alone_updates: int = 40
    rookie_updates: int = 160
    checkpoint_every: int = 5
    # the game
    encoder: int = 2
    fps: int = 8
    num_envs: int = 16
    # PPO
    num_steps: int = 128  # ticks a rollout plays in each match
    hidden: tuple[int, ...] = (64, 64)
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    epochs: int = 4
    minibatches: int = 4
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    # Glutton reward
    food_weight: float = 1.0
    tick_bonus: float = 0.1
    threads: int = 1  # torch's threads; one keeps a run exactly repeatable and leaves cores to the engine

    def check(self) -> None:
        if self.personality not in REWARDS:
            raise ValueError(f"personality must be one of {sorted(REWARDS)}, not {self.personality!r}")
        if self.encoder not in ENCODER_SIZES:
            raise ValueError("encoder must be 1 or 2")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be in [0, 2**32)")
        if self.checkpoint_every < 1 or self.alone_updates < 0 or self.rookie_updates < 0:
            raise ValueError("checkpoint_every must be at least 1 and the stages' updates 0 or more")
        if self.alone_updates % self.checkpoint_every or self.rookie_updates % self.checkpoint_every:
            raise ValueError("each stage must be a whole number of checkpoints (multiples of checkpoint_every)")
        if self.alone_updates + self.rookie_updates == 0:
            raise ValueError("nothing to train: both stages are 0 updates")
        if self.num_envs * self.num_steps < self.minibatches or self.minibatches < 1:
            raise ValueError("a rollout needs at least one tick a minibatch")

    @property
    def checkpoints(self) -> int:
        return (self.alone_updates + self.rookie_updates) // self.checkpoint_every

    def stage(self, checkpoint: int) -> str:
        return "alone" if checkpoint * self.checkpoint_every < self.alone_updates else "rookie"

    def to_json(self) -> dict:
        """The file's fields: the shared ones the dashboard and `promote` read, then the PPO ones as they are."""
        shared = {
            "method": "ppo",
            "personality": self.personality,
            "seed": self.seed,
            "generations": self.checkpoints,
            "aloneGenerations": self.alone_updates // self.checkpoint_every,
            "rookieGenerations": self.rookie_updates // self.checkpoint_every,
            "fps": self.fps,
        }
        return {**shared, **{k: v for k, v in asdict(self).items() if k not in shared}}

    @classmethod
    def from_json(cls, data: dict) -> "Settings":
        known = {f.name for f in fields(cls)}
        picked = {k: v for k, v in data.items() if k in known}
        if "hidden" in picked:
            picked["hidden"] = tuple(picked["hidden"])
        return cls(**picked)


# The run folder -----------------------------------------------------------------------------


def _write_whole(path: Path, data: bytes | str) -> None:
    """Writes so that `path` is always either its old contents or all of `data`."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data.encode() if isinstance(data, str) else data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _json(value) -> str:
    return json.dumps(value, indent=2) + "\n"


def run_files(run: Path) -> dict[str, Path]:
    return {
        "settings": run / "settings.json",
        "log": run / "log.jsonl",
        "checkpoint": run / "checkpoint.pt",
        "best": run / "best.json",
        "generations": run / "generations",
    }


def save_checkpoint(run: Path, settings: Settings, agent: Agent, optimizer, line: dict, brain: dict) -> None:
    """A checkpoint's files: its brain, the log line, then the state to resume from, last."""
    files = run_files(run)
    _write_whole(files["generations"] / f"gen-{line['generation']:04d}.json", _json(brain))
    _write_whole(files["best"], _json(brain))
    with open(files["log"], "a") as log:
        log.write(json.dumps(line) + "\n")
        log.flush()
        os.fsync(log.fileno())
    tmp = files["checkpoint"].with_name("checkpoint.pt.tmp")
    torch.save({"generation": line["generation"] + 1, "agent": agent.state_dict(), "optimizer": optimizer.state_dict()}, tmp)
    os.replace(tmp, files["checkpoint"])


def read_log(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


# Training -----------------------------------------------------------------------------------


def _seed_for(settings: Settings, checkpoint: int) -> int:
    return (settings.seed * 1_000_003 + checkpoint * 7_919 + 1) % 2**32


def make_env(settings: Settings, stage: str) -> SnakeVectorEnv:
    """Matches mixing both modes and delays 0 to 4 (the learner's and the rookie's separately); alone on the board, or 1v1."""
    rookie = stage == "rookie"
    return SnakeVectorEnv(
        settings.num_envs,
        encoder=settings.encoder,
        opponents=["rookie"] if rookie else [],
        modes=("timed", "endless"),
        learner_delay=(0, 4),
        opponent_delay=(0, 4),
        seat="random" if rookie else 0,
        fps=settings.fps,
    )


def train(run: Path, settings: Settings, *, stop=lambda: False, echo=lambda text: print(text, flush=True)) -> int:
    """Trains a run folder made by `create_run`, from its checkpoint if it has one. Returns the next checkpoint number."""
    files = run_files(run)
    torch.set_num_threads(settings.threads)
    torch.manual_seed(settings.seed)
    agent = Agent(ENCODER_SIZES[settings.encoder], settings.hidden)
    optimizer = torch.optim.Adam(agent.parameters(), lr=settings.learning_rate, eps=1e-5)
    start = 0
    if files["checkpoint"].exists():
        saved = torch.load(files["checkpoint"], weights_only=True)
        agent.load_state_dict(saved["agent"])
        optimizer.load_state_dict(saved["optimizer"])
        start = saved["generation"]
    # Drop log lines past the checkpoint (a crash between the log and the checkpoint), as the trainer does.
    log = read_log(files["log"])[:start]
    _write_whole(files["log"], "".join(json.dumps(line) + "\n" for line in log))

    reward_fn = lambda facts: REWARDS[settings.personality](facts, asdict(settings))  # noqa: E731
    generator = torch.Generator()
    env, env_stage = None, None
    try:
        for checkpoint in range(start, settings.checkpoints):
            if stop():
                break
            stage = settings.stage(checkpoint)
            if stage != env_stage:
                if env is not None:
                    env.close()
                env, env_stage = make_env(settings, stage), stage
            obs, _ = env.reset(seed=_seed_for(settings, checkpoint))
            generator.manual_seed(_seed_for(settings, checkpoint))
            episode_return = np.zeros(settings.num_envs, dtype=np.float32)
            returns: list[float] = []
            stats: dict[str, float] = {}
            for _ in range(settings.checkpoint_every):
                rollout = collect(agent, env, reward_fn, obs, episode_return, settings.num_steps, generator)
                obs = rollout.last_obs.numpy()
                returns += rollout.finished_returns
                stats = update(
                    agent, optimizer, rollout, generator,
                    gamma=settings.gamma, gae_lambda=settings.gae_lambda, epochs=settings.epochs,
                    minibatches=settings.minibatches, clip=settings.clip, vf_coef=settings.vf_coef,
                    ent_coef=settings.ent_coef, max_grad_norm=settings.max_grad_norm,
                )
            # No episode ended this stretch (very long rounds): repeat the last figures, and `episodes` says so.
            best = max(returns) if returns else (log[-1]["best"] if log else 0.0)
            mean = float(np.mean(returns)) if returns else (log[-1]["mean"] if log else 0.0)
            line = {
                "generation": checkpoint, "stage": stage, "best": best, "mean": mean,
                "update": (checkpoint + 1) * settings.checkpoint_every,
                "steps": (checkpoint + 1) * settings.checkpoint_every * settings.num_envs * settings.num_steps,
                "episodes": len(returns), **stats,
            }
            log.append(line)
            save_checkpoint(run, settings, agent, optimizer, line, export_brain(agent.policy, settings.encoder))
            echo(json.dumps(line))
            start = checkpoint + 1
    finally:
        if env is not None:
            env.close()
    return start


def create_run(run: Path, settings: Settings) -> None:
    settings.check()
    if run.exists():
        raise FileExistsError(f"{run} already exists: name a new run folder, or --resume it")
    run_files(run)["generations"].mkdir(parents=True)
    _write_whole(run_files(run)["settings"], _json(settings.to_json()))


def open_run(run: Path, rookie_updates: int | None = None) -> Settings:
    """A run folder's settings. `rookie_updates` may be raised (or lowered) to carry on further, and is saved."""
    path = run_files(run)["settings"]
    if not path.exists():
        raise FileNotFoundError(f"{run} isn't a run folder: it has no settings.json")
    data = json.loads(path.read_text())
    if data.get("method") != "ppo":
        raise ValueError(f"{run} isn't a PPO run (method {data.get('method')!r})")
    settings = Settings.from_json(data)
    if rookie_updates is not None:
        settings.rookie_updates = rookie_updates
    settings.check()
    if rookie_updates is not None:
        _write_whole(path, _json(settings.to_json()))
    return settings


def main(argv: list[str] | None = None) -> int:
    defaults = Settings()
    p = argparse.ArgumentParser(prog="snake_lab.run", description="Trains a snake policy with PPO.")
    p.add_argument("--run", type=Path, help="the new run folder (must not exist)")
    p.add_argument("--resume", type=Path, help="carry on a run from its latest checkpoint")
    for f in fields(Settings):
        if f.name == "hidden":
            p.add_argument("--hidden", type=int, nargs="+", help=f"hidden layer sizes (default {list(defaults.hidden)})")
        else:
            p.add_argument(f"--{f.name.replace('_', '-')}", type=type(getattr(defaults, f.name)), help=f"default {getattr(defaults, f.name)}" + (" (on --resume: a new length, to train further)" if f.name == "rookie_updates" else ""))
    args = p.parse_args(argv)

    if args.resume:
        given = [f.name for f in fields(Settings) if getattr(args, f.name, None) is not None and f.name != "rookie_updates"]
        if args.run or given:
            p.error("--resume takes only --rookie-updates")
        run, settings = args.resume, open_run(args.resume, args.rookie_updates)
    else:
        if not args.run:
            p.error("give --run (a new folder) or --resume")
        picked = {f.name: getattr(args, f.name) for f in fields(Settings) if getattr(args, f.name, None) is not None}
        if args.rookie_updates is not None:
            picked["rookie_updates"] = args.rookie_updates
        if "hidden" in picked:
            picked["hidden"] = tuple(picked["hidden"])
        run, settings = args.run, Settings(**picked)
        create_run(run, settings)

    # Ctrl-C or SIGTERM: the checkpoint in progress finishes and is saved, then we stop. A second one abandons it.
    stopping = []

    def on_signal(signum, _frame):
        if stopping:
            os._exit(130)
        stopping.append(signum)
        print("Stopping after this checkpoint (again to abandon it)...", file=sys.stderr, flush=True)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    done = train(run, settings, stop=lambda: bool(stopping))
    print(f"{'Stopped' if stopping else 'Finished'}: {done} of {settings.checkpoints} checkpoints saved in {run}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
