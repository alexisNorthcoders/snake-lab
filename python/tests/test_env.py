import sys

import numpy as np
import pytest

from snake_lab import FACT_FIELDS, EnvError, EnvProcessError, SnakeVectorEnv


@pytest.fixture
def make():
    envs = []

    def factory(*args, **kwargs):
        env = SnakeVectorEnv(*args, **kwargs)
        envs.append(env)
        return env

    yield factory
    for env in envs:
        env.close()


@pytest.mark.parametrize("encoder,size", [(1, 23), (2, 328)])
def test_reset_and_step_shapes_and_types(make, encoder, size):
    env = make(3, encoder=encoder, opponents=["rookie"])
    obs, infos = env.reset(seed=1)
    assert obs.shape == (3, size) and obs.dtype == np.float32
    assert set(infos) == set(FACT_FIELDS)
    assert infos["alive"].tolist() == [1, 1, 1]
    obs, reward, terminated, truncated, infos = env.step(np.ones(3, dtype=np.int64))
    assert obs.shape == (3, size) and obs.dtype == np.float32
    assert reward.shape == (3,) and not reward.any()
    assert terminated.shape == truncated.shape == (3,)
    assert terminated.dtype == truncated.dtype == np.bool_
    assert infos["ticks"].tolist() == [1, 1, 1]
    assert infos["ticks"].dtype == np.int32
    assert env.observation_space.contains(obs)


def rollout(env, seed, steps=60):
    rng = np.random.default_rng(99)
    obs, infos = env.reset(seed=seed)
    trace = [(obs, infos)]
    for _ in range(steps):
        obs, _, term, trunc, infos = env.step(rng.integers(0, 3, env.num_envs))
        trace.append((obs, infos, term, trunc))
    return trace


def test_same_seed_and_actions_are_identical(make):
    kwargs = dict(opponents=["rookie"], seat="random", learner_delay=[0, 4])
    a = rollout(make(4, **kwargs), 7)
    b = rollout(make(4, **kwargs), 7)
    for x, y in zip(a, b):
        assert np.array_equal(x[0], y[0])
        assert all(np.array_equal(x[1][k], y[1][k]) for k in FACT_FIELDS)
        assert all(np.array_equal(p, q) for p, q in zip(x[2:], y[2:]))
    c = rollout(make(4, **kwargs), 8)
    assert not np.array_equal(a[-1][0], c[-1][0])


def test_a_match_that_ends_is_flagged_and_restarts(make):
    env = make(2, encoder=1, modes="timed")
    env.reset(seed=3)
    for _ in range(3000):  # driving straight, a lone snake ends its round by dying or by time
        obs, _, terminated, truncated, infos = env.step(np.ones(2, dtype=np.int64))
        if (terminated | truncated).any():
            break
    else:
        pytest.fail("no match ended")
    i = int(np.argmax(terminated | truncated))
    assert infos["ended"][i] == 1 and infos["outcome"][i] in (1, 2, 3) and infos["reason"][i] in (1, 2, 3)
    assert infos["ticks"][i] > 1
    _, _, _, _, after = env.step(np.ones(2, dtype=np.int64))
    assert after["ticks"][i] == 1 and after["ended"][i] == 0


def test_timeout_is_truncation_and_death_is_termination(make):
    env = make(1, encoder=1, modes="timed", end_on_death=False)
    env.reset(seed=1)
    while True:
        _, _, terminated, truncated, infos = env.step([1])
        if infos["ended"][0]:
            break
    assert terminated[0] != truncated[0]
    assert truncated[0] == (infos["reason"][0] == 2)


def test_bad_actions_raise_and_env_survives(make):
    env = make(2, encoder=1)
    env.reset(seed=1)
    with pytest.raises(ValueError):
        env.step([1, 3])
    with pytest.raises(ValueError):
        env.step([1])
    env.step([1, 1])


def test_engine_errors_come_through_as_exceptions(make):
    env = make(1, opponents=["nobody-by-that-name"])
    with pytest.raises(EnvError, match="nobody-by-that-name"):
        env.reset(seed=1)


def test_step_before_reset_is_an_error(make):
    env = make(1)
    with pytest.raises(EnvError):
        env.step([1])


def test_a_subprocess_that_dies_raises(make):
    env = make(2)
    env.reset(seed=1)
    env._proc.kill()
    with pytest.raises(EnvProcessError):
        env.step([1, 1])
    with pytest.raises(EnvProcessError):
        env.step([1, 1])  # closed now: still an error, never a hang


def test_a_subprocess_that_exits_at_once_raises():
    env = SnakeVectorEnv(1, command=[sys.executable, "-c", "import sys; print('boom', file=sys.stderr); sys.exit(3)"])
    with pytest.raises(EnvProcessError, match="boom"):
        env.reset(seed=1)


def test_close_stops_the_subprocess(make):
    env = make(1)
    proc = env._proc
    env.reset(seed=1)
    env.close()
    assert proc.poll() == 0
