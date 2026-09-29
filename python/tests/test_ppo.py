import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from snake_lab.brain import export_brain
from snake_lab.env import REPO_ROOT
from snake_lab.ppo import Agent, compute_gae, ppo_loss
from snake_lab.rewards import REWARDS, glutton_reward
from snake_lab.run import Settings, create_run, open_run, read_log, run_files, train


def t(*values):
    return torch.tensor(values, dtype=torch.float32)


# Advantages ---------------------------------------------------------------------------------


def test_gae_worked_by_hand():
    # gamma 0.5, lambda 0.5, one match, three ticks. Backwards from the last:
    #   t=2: delta = 2 + 0.5*4 - 0 = 4,            adv = 4
    #   t=1: delta = 0 + 0.5*0 - 1 = -1,           adv = -1 + 0.25*4 = 0
    #   t=0: delta = 1 + 0.5*1 - 0.5 = 1,          adv = 1 + 0.25*0 = 1
    rewards, values = t(1, 0, 2).unsqueeze(1), t(0.5, 1, 0).unsqueeze(1)
    advantages, returns = compute_gae(rewards, values, torch.zeros(3, 1), t(4), gamma=0.5, lam=0.5)
    assert advantages.squeeze().tolist() == pytest.approx([1, 0, 4])
    assert returns.squeeze().tolist() == pytest.approx([1.5, 1, 4])  # advantage + value


def test_gae_stops_at_the_end_of_an_episode():
    # the episode ends on tick 1: its next state's value and later advantages don't count towards it
    #   t=2: adv 4 (as before); t=1: delta = 0 - 1 = -1, adv = -1; t=0: delta = 1 + 0.5*1 - 0.5 = 1, adv = 1 + 0.25*(-1)
    rewards, values = t(1, 0, 2).unsqueeze(1), t(0.5, 1, 0).unsqueeze(1)
    advantages, _ = compute_gae(rewards, values, t(0, 1, 0).unsqueeze(1), t(4), gamma=0.5, lam=0.5)
    assert advantages.squeeze().tolist() == pytest.approx([0.75, -1, 4])


def test_gae_matches_each_match_separately():
    rewards, values = t(1, 0, 2), t(0.5, 1, 0)
    both, _ = compute_gae(torch.stack([rewards, rewards * 2], 1), torch.stack([values, values], 1), torch.zeros(3, 2), t(4, 4), 0.5, 0.5)
    alone, _ = compute_gae(rewards.unsqueeze(1), values.unsqueeze(1), torch.zeros(3, 1), t(4), 0.5, 0.5)
    assert both[:, 0].tolist() == pytest.approx(alone.squeeze().tolist())


# The loss -----------------------------------------------------------------------------------


def test_ppo_loss_worked_by_hand():
    # ratios 1.5 and 0.5 (old log-probs 0), advantages 1 and -1, clip 0.2
    #   tick 0: max(-1*1.5, -1*1.2) = -1.2     tick 1: max(1*0.5, 1*0.8) = 0.8    policy loss = -0.2
    #   value loss = 0.5 * mean((1-0)^2, (3-1)^2) = 1.25       entropy = 0.75
    #   total = -0.2 + 0.5*1.25 - 0.01*0.75 = 0.4175
    loss = ppo_loss(t(1.5, 0.5).log(), t(0, 0), t(1, -1), t(1, 3), t(0, 1), t(1.0, 0.5), clip=0.2, vf_coef=0.5, ent_coef=0.01)
    assert float(loss.policy) == pytest.approx(-0.2)
    assert float(loss.value) == pytest.approx(1.25)
    assert float(loss.entropy) == pytest.approx(0.75)
    assert float(loss.total) == pytest.approx(0.4175)
    assert float(loss.clip_fraction) == pytest.approx(1.0)
    assert float(loss.approx_kl) == pytest.approx(((0.5 - np.log(1.5)) + (-0.5 - np.log(0.5))) / 2)


def test_ppo_loss_clipping_stops_the_gradient():
    # a good action (advantage 1) already made 1.5x likelier is past the clip: no more push on it
    log_prob = t(1.5).log().requires_grad_()
    ppo_loss(log_prob, t(0), t(1), t(0), t(0), t(0), clip=0.2).policy.backward()
    assert float(log_prob.grad) == 0.0
    # inside the clip, the push is -advantage * ratio
    log_prob = t(1.1).log().requires_grad_()
    ppo_loss(log_prob, t(0), t(1), t(0), t(0), t(0), clip=0.2).policy.backward()
    assert float(log_prob.grad) == pytest.approx(-1.1, rel=1e-5)


# The reward ---------------------------------------------------------------------------------


def facts(**fields):
    n = len(next(iter(fields.values())))
    base = {"scoreGained": [0] * n, "alive": [1] * n}
    return {k: np.array(v, dtype=np.int32) for k, v in {**base, **fields}.items()}


def test_glutton_reward_from_facts():
    f = facts(scoreGained=[0, 10, 20, -5, 0], alive=[1, 1, 0, 1, 0])
    assert glutton_reward(f).tolist() == pytest.approx([0.1, 10.1, 20.0, -4.9, 0.0])
    assert glutton_reward(f, food_weight=2, tick_bonus=0).tolist() == pytest.approx([0, 20, 40, -10, 0])
    assert glutton_reward(f).dtype == np.float32


def test_rewards_are_chosen_by_personality_with_the_settings_weights():
    settings = {"food_weight": 3.0, "tick_bonus": 0.5}
    assert REWARDS["glutton"](facts(scoreGained=[1], alive=[1]), settings).tolist() == [3.5]


# The export ---------------------------------------------------------------------------------


def node_says(brain, inputs):
    proc = subprocess.run(
        ["node", "--import", "tsx", "python/tests/brain_check.ts"],
        input=json.dumps({"brain": brain, "inputs": inputs}), capture_output=True, text=True, cwd=REPO_ROOT, timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


@pytest.mark.parametrize("encoder,size,hidden", [(2, 328, (64, 64)), (1, 23, (16,))])
def test_exported_brain_is_valid_and_agrees_with_the_policy(encoder, size, hidden):
    torch.manual_seed(3)
    agent = Agent(size, hidden)
    for p in agent.policy.parameters():  # a trained-looking policy: not the near-uniform one it starts as
        torch.nn.init.normal_(p, std=0.5)
    brain = export_brain(agent.policy, encoder)
    assert brain["sizes"] == [size, *hidden, 3] and brain["encoderVersion"] == encoder

    obs = torch.randn(200, size)
    said = node_says(brain, obs.tolist())
    assert said["problems"] == []
    assert brain["rulesVersion"] == said["rulesVersion"]
    logits = agent.policy(obs).detach()
    assert np.array(said["outputs"]) == pytest.approx(logits.numpy(), abs=1e-4)
    assert np.argmax(said["outputs"], axis=1).tolist() == logits.argmax(1).tolist()


# A run --------------------------------------------------------------------------------------


def tiny(**overrides) -> Settings:
    return Settings(**{
        "seed": 5, "alone_updates": 2, "rookie_updates": 2, "checkpoint_every": 1, "num_envs": 2, "num_steps": 16,
        "hidden": (8,), "minibatches": 2, "epochs": 1, **overrides,
    })


def test_a_run_writes_the_shared_layout(tmp_path):
    run = tmp_path / "run"
    settings = tiny()
    create_run(run, settings)
    assert train(run, settings, echo=lambda _: None) == 4
    files = run_files(run)

    saved = json.loads(files["settings"].read_text())
    assert saved["method"] == "ppo" and saved["personality"] == "glutton" and saved["seed"] == 5
    assert saved["generations"] == 4 and saved["aloneGenerations"] == 2 and saved["rookieGenerations"] == 2

    log = read_log(files["log"])
    assert [line["generation"] for line in log] == [0, 1, 2, 3]
    assert [line["stage"] for line in log] == ["alone", "alone", "rookie", "rookie"]
    assert all({"best", "mean", "episodes", "entropy", "policy_loss"} <= set(line) for line in log)
    assert sorted(p.name for p in files["generations"].iterdir()) == [f"gen-000{g}.json" for g in range(4)]
    assert json.loads(files["best"].read_text()) == json.loads((files["generations"] / "gen-0003.json").read_text())
    assert files["checkpoint"].exists() and not list(run.rglob("*.tmp"))
    assert node_says(json.loads(files["best"].read_text()), [[0.0] * 328])["problems"] == []


def test_a_resumed_run_is_the_run_that_never_stopped(tmp_path):
    settings = tiny(alone_updates=4, rookie_updates=0)
    whole, split = tmp_path / "whole", tmp_path / "split"
    create_run(whole, settings)
    train(whole, settings, echo=lambda _: None)

    create_run(split, settings)
    calls = []

    def stop_after_two():
        calls.append(1)
        return len(calls) > 2

    assert train(split, settings, stop=stop_after_two, echo=lambda _: None) == 2
    assert len(read_log(run_files(split)["log"])) == 2
    assert train(split, open_run(split), echo=lambda _: None) == 4

    assert read_log(run_files(split)["log"]) == read_log(run_files(whole)["log"])
    for g in range(4):
        name = f"gen-{g:04d}.json"
        assert (run_files(split)["generations"] / name).read_text() == (run_files(whole)["generations"] / name).read_text()


def test_resume_can_lengthen_the_rookie_stage_and_refuses_other_runs(tmp_path):
    run = tmp_path / "run"
    create_run(run, tiny(alone_updates=1, rookie_updates=1))
    assert open_run(run, rookie_updates=3).checkpoints == 4
    assert json.loads(run_files(run)["settings"].read_text())["generations"] == 4
    with pytest.raises(FileExistsError):
        create_run(run, tiny())
    (run / "settings.json").write_text(json.dumps({"method": "neuroevolution"}))
    with pytest.raises(ValueError, match="isn't a PPO run"):
        open_run(run)


def test_settings_are_checked():
    with pytest.raises(ValueError, match="whole number of checkpoints"):
        tiny(checkpoint_every=3).check()
    with pytest.raises(ValueError, match="personality"):
        tiny(personality="nope").check()
