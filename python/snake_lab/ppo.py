"""PPO (proximal policy optimisation), written out so each step can be followed. After CleanRL's `ppo.py`.

The loop, which `run.py` drives:

1. **Roll out**: play `num_steps` ticks in every match of a batch with the current policy, keeping each tick's
   observation, action, the log-probability the policy gave that action, the reward, whether the episode ended, and
   the value network's guess of how good the state was.
2. **Advantages** (`compute_gae`): for each tick, how much better the action turned out than the value network
   expected, looking a few ticks ahead and fading (generalised advantage estimation).
3. **Update** (`update`): a few passes over the rollout in shuffled minibatches, nudging the policy so good actions
   become likelier, but not by more than a clip allows (`ppo_loss`), and the value network towards the returns.

Two separate networks: the policy (observation to left, straight, right logits) and the value network (observation
to one number). Only the policy is exported (`brain.py`).
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn


def mlp(sizes: list[int], out_std: float) -> nn.Sequential:
    """A fully connected network with `tanh` between layers and a linear last layer, orthogonally initialised."""
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        last = i == len(sizes) - 2
        linear = nn.Linear(sizes[i], sizes[i + 1])
        nn.init.orthogonal_(linear.weight, out_std if last else float(np.sqrt(2)))
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        if not last:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class Agent(nn.Module):
    """A policy network and a separate value network over the same observations."""

    def __init__(self, obs_size: int, hidden: tuple[int, ...] = (64, 64), actions: int = 3) -> None:
        super().__init__()
        self.policy = mlp([obs_size, *hidden, actions], out_std=0.01)  # small: start close to uniform
        self.value = mlp([obs_size, *hidden, 1], out_std=1.0)

    def act(self, obs: torch.Tensor, generator: torch.Generator | None = None, action: torch.Tensor | None = None):
        """Samples an action (or scores the given one): the action, its log-probability, the entropy and the value."""
        log_probs = torch.log_softmax(self.policy(obs), dim=-1)
        if action is None:
            action = torch.multinomial(log_probs.exp(), 1, generator=generator).squeeze(-1)
        log_prob = log_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        entropy = -(log_probs.exp() * log_probs).sum(-1)
        return action, log_prob, entropy, self.value(obs).squeeze(-1)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalised advantage estimation. Returns `(advantages, returns)`.

    All are `[steps, envs]`. `dones[t]` is 1 if the episode ended on tick `t` (so the state after it is a new
    episode's, and nothing after it counts towards `t`). `last_value` is the value of the state after the last tick.

    The one-tick error is `delta[t] = rewards[t] + gamma * value[t+1] - values[t]`: what happened, plus what the
    next state is worth, against what the state was thought to be worth. The advantage adds up the deltas ahead,
    fading by `gamma * lam` each tick: `adv[t] = delta[t] + gamma * lam * adv[t+1]`, worked backwards from the end.
    The return the value network learns is `advantage + value`.
    """
    steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(last_value)
    for t in reversed(range(steps)):
        next_value = last_value if t == steps - 1 else values[t + 1]
        alive = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * alive - values[t]
        running = delta + gamma * lam * alive * running
        advantages[t] = running
    return advantages, advantages + values


@dataclass
class LossParts:
    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor


def ppo_loss(
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    new_value: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    clip: float = 0.2,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
) -> LossParts:
    """The PPO loss on a minibatch: clipped surrogate, value error and an entropy bonus.

    - `ratio` is how much likelier the updated policy makes the action than the one that played it. Multiplied by
      the advantage it says "make good actions likelier". The clipped surrogate takes the pessimistic one of that and
      the same with the ratio held within `1 ± clip`, so a step gains nothing by moving the policy further than that.
    - The value loss is half the mean squared error of the value network against the returns.
    - The entropy bonus (subtracted, since we minimise) keeps the policy from settling too soon on one move.
    """
    log_ratio = new_log_prob - old_log_prob
    ratio = log_ratio.exp()
    unclipped = -advantages * ratio
    clipped = -advantages * torch.clamp(ratio, 1 - clip, 1 + clip)
    policy_loss = torch.max(unclipped, clipped).mean()
    value_loss = 0.5 * ((new_value - returns) ** 2).mean()
    entropy_mean = entropy.mean()
    with torch.no_grad():
        approx_kl = ((ratio - 1) - log_ratio).mean()  # an estimate of how far the policy moved
        clip_fraction = ((ratio - 1).abs() > clip).float().mean()
    total = policy_loss + vf_coef * value_loss - ent_coef * entropy_mean
    return LossParts(total, policy_loss, value_loss, entropy_mean, approx_kl, clip_fraction)


@dataclass
class Rollout:
    obs: torch.Tensor  # [steps, envs, obs]
    actions: torch.Tensor  # [steps, envs]
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    last_obs: torch.Tensor  # the observation after the last tick
    finished_returns: list[float]  # return of each episode that ended in the rollout


def collect(
    agent: Agent,
    env,
    reward_fn: Callable[[dict], np.ndarray],
    obs: np.ndarray,
    episode_return: np.ndarray,
    num_steps: int,
    generator: torch.Generator,
) -> Rollout:
    """Plays `num_steps` ticks in every match of `env` from `obs`, and updates `episode_return` (running, per match).

    A match that ends restarts inside the env with its next observation already given, so the loop never resets.
    An episode that ran out of time is treated as ended: the env doesn't hand back its last observation to bootstrap
    from, a small bias that only touches timed rounds' final tick.
    """
    n = obs.shape[0]
    size = obs.shape[1]
    buf_obs = torch.zeros(num_steps, n, size)
    buf_actions = torch.zeros(num_steps, n, dtype=torch.long)
    buf_log_probs = torch.zeros(num_steps, n)
    buf_rewards = torch.zeros(num_steps, n)
    buf_dones = torch.zeros(num_steps, n)
    buf_values = torch.zeros(num_steps, n)
    finished: list[float] = []
    current = torch.as_tensor(obs)
    for t in range(num_steps):
        with torch.no_grad():
            action, log_prob, _, value = agent.act(current, generator)
        next_obs, _, terminated, truncated, infos = env.step(action.numpy())
        reward = reward_fn(infos)
        done = terminated | truncated
        buf_obs[t], buf_actions[t], buf_log_probs[t], buf_values[t] = current, action, log_prob, value
        buf_rewards[t] = torch.as_tensor(reward)
        buf_dones[t] = torch.as_tensor(done.astype(np.float32))
        episode_return += reward
        for i in np.flatnonzero(done):
            finished.append(float(episode_return[i]))
            episode_return[i] = 0.0
        current = torch.as_tensor(next_obs)
    return Rollout(buf_obs, buf_actions, buf_log_probs, buf_rewards, buf_dones, buf_values, current, finished)


def update(
    agent: Agent,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    generator: torch.Generator,
    *,
    gamma: float,
    gae_lambda: float,
    epochs: int,
    minibatches: int,
    clip: float,
    vf_coef: float,
    ent_coef: float,
    max_grad_norm: float,
) -> dict[str, float]:
    """One PPO update from a rollout: advantages, then `epochs` passes of shuffled minibatches. Returns the mean losses."""
    with torch.no_grad():
        last_value = agent.value(rollout.last_obs).squeeze(-1)
    advantages, returns = compute_gae(rollout.rewards, rollout.values, rollout.dones, last_value, gamma, gae_lambda)

    # Flatten [steps, envs] into one batch of ticks.
    obs = rollout.obs.reshape(-1, rollout.obs.shape[-1])
    actions = rollout.actions.reshape(-1)
    old_log_probs = rollout.log_probs.reshape(-1)
    advantages = advantages.reshape(-1)
    returns = returns.reshape(-1)

    batch = obs.shape[0]
    size = batch // minibatches
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0}
    passes = 0
    for _ in range(epochs):
        order = torch.randperm(batch, generator=generator)
        for start in range(0, batch - size + 1, size):
            idx = order[start : start + size]
            _, new_log_prob, entropy, new_value = agent.act(obs[idx], action=actions[idx])
            adv = advantages[idx]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # normalised a minibatch at a time
            loss = ppo_loss(new_log_prob, old_log_probs[idx], adv, new_value, returns[idx], entropy, clip, vf_coef, ent_coef)
            optimizer.zero_grad()
            loss.total.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()
            for key, part in (("policy_loss", loss.policy), ("value_loss", loss.value), ("entropy", loss.entropy),
                              ("approx_kl", loss.approx_kl), ("clip_fraction", loss.clip_fraction)):
                stats[key] += float(part.detach())
            passes += 1
    return {key: value / passes for key, value in stats.items()}
