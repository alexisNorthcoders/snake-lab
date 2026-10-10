# snake-lab

Training ground for the snake game's AI opponents: headless matches, the evaluation gauntlet,
neuroevolution and (later) reinforcement learning.

The game's rules come from [`snake-colyseus`](https://github.com/alexisNorthcoders/snake-colyseus)'s
engine, installed as a git dependency pinned to an `engine-vX.Y.Z` tag. Nothing here copies the rules.

## Running a match

```sh
npm install
npm run match -- --seed 7 --players rookie,dummy --mode timed --fps 8 --delays 2
```

This plays one round, headless and as fast as it can, and prints why it ended, who won, and each
snake's score, length and death. The same options always play the same match, so any match can be
replayed from its seed. `--events` also lists every pellet eaten and every death, tick by tick,
and `--json` prints the whole result. Players are roster ids (`rookie`, `dummy`) or brain files, seated in the order
given (1 to 4), and `--delays` takes one reaction delay for everyone or one per player.

## Running the gauntlet

```sh
npm run gauntlet -- rookie              # a roster id
npm run gauntlet -- path/to/brain.json  # or a brain file, checked first
```

The gauntlet decides whether a snake is good enough to join the roster. It plays the candidate in
1v1 matches against the rookie and every other roster snake, at each reaction delay from 0 to 4 and
in both modes. Each seed is played twice, once with the candidate in each seat, and both snakes
play at the same delay. It prints the win, loss and draw rates for each opponent, delay and mode.
A draw counts as not winning. It passes a candidate that wins at least 60% of its matches against
the rookie at delay 2, over both modes, and exits 0 on a pass and 1 on a fail. `--seeds` (default
100 per cell), `--base-seed` (default 1) and `--fps` (default 8) change the run, and the same
options always give the same report. The first numbers are in
[`docs/benchmarks/`](docs/benchmarks/2026-09-28-first-gauntlet.md).

## Training a snake

```sh
npm run train -- --personality glutton --run runs/glutton-1 --seed 1
```

This trains a population of brains by neuroevolution. Each generation, every snake plays the same
seeded matches (both modes, reaction delays 0 to 4) and is scored by its fitness: for a Glutton, its
food score plus a small bonus for every tick it survives and a bigger one for every match it wins,
averaged over its matches. The fittest are
kept as they are, and the rest of the next generation is bred from the fittest by crossover and
mutation. The first generations play alone on the board, then against the rookie, then in the league against varied opponents (some four-player). It trains a Glutton, Survivor or Hunter. It prints the
best and mean fitness each generation, and at the end how many matches a second and generations an
hour it played. The same seed and settings always give the same run.

The run folder keeps every generation's best brain (`generations/gen-0010.json` and so on), so any
generation can be tried or promoted later, the latest again as `best.json`, and a checkpoint of the
latest population. Ctrl-C stops the run after the current generation, and

```sh
npm run train -- --resume runs/glutton-1 [--generations 500]
```

carries on from the checkpoint, with exactly the result of a run that never stopped. Matches are
played on every core (`--workers` to choose), which never changes the result. A brain can be tried
with `npm run match -- --players runs/glutton-1/best.json,rookie` and put through the gauntlet.
`--help` lists the settings (population, mutation, stages and so on); only the Glutton can be
trained for now. `AGENTS.md` describes running a long job in the background.

## Promoting a snake

```sh
npm run promote -- runs/glutton-1 50 --name "Nimble Pete" [--target ../snake-colyseus]
```

This puts generation 50's best brain through the gauntlet (`--seeds`, `--base-seed` and `--fps`
work as there). If it passes, it writes the brain and a roster entry into the `snake-colyseus`
checkout, bumps that repo's `engineVersion` by a minor version, and prints a PR description with
the training settings, the gauntlet's table and verdict, the command and the engine tag; it's saved
in the run folder as `promote-<id>.md`. If it fails, or the name or its id is already in the
roster, or the target isn't a `snake-colyseus` checkout, or the generation's brain is missing,
it writes nothing and exits 1. It never runs git. What's left is for a person:

1. In `snake-colyseus`, branch, commit the brain, the entry and `package.json`, and open the PR
   with the printed description.
2. When it's merged and tagged, bump the `snake-colyseus` pin in this repo's `package.json` to the
   new `engine-vX.Y.Z` and `npm install`.

## The training dashboard

```sh
npm run dashboard -- runs/glutton-1 --port 8091 [--host 127.0.0.1]   # 8080 is the default, but go-server uses it
```

Serves a page next to a run: its settings, the latest generation and stage, and a chart of best and
mean fitness per generation with the stages shaded. A run that's training grows on the chart as each
generation lands, with no reload; a finished or stopped one shows complete. It only reads the run
folder. Under the chart, a grid replays a generation's best snake in that generation's own fixtures, and
beside it the **network** of that snake is drawn: the 23 labelled inputs, the hidden units and left,
straight and right, edges blue or orange by the sign of their weight and thicker the bigger it is (the
faintest are left out), nodes shaded by their value. It lights up tick by tick with the followed game
(the first, or click another) and marks the output that won. It listens on `127.0.0.1`: from another machine use `ssh -L 8080:127.0.0.1:8080 <box>` and open
`http://localhost:8080`. Never expose it publicly (`--host` is for the Pi's LAN).

## The Python environment

`python/` holds a Gymnasium vector environment over the engine, for the PPO trainer. Set it up (after `npm install`) and test it:

```sh
cd python
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/python -m pytest
.venv/bin/python -m snake_lab.bench    # environment steps a second, per batch size
```

```python
from snake_lab import SnakeVectorEnv
env = SnakeVectorEnv(64, encoder=2, opponents=["rookie"], learner_delay=[0, 4])
obs, infos = env.reset(seed=1)
obs, reward, terminated, truncated, infos = env.step(actions)  # actions: 0 left, 1 straight, 2 right
env.close()
```

## Training with PPO

The PPO trainer in `python/` learns a Glutton through that environment: alone on the board, then 1v1 against the rookie. It writes the same run folder as `npm run train`, so the dashboard, `npm run match`, `npm run gauntlet` and `npm run promote` all read it.

```sh
cd python
.venv/bin/python -m snake_lab.run --personality glutton --run ../runs/ppo-1 --seed 1
.venv/bin/python -m snake_lab.run --resume ../runs/ppo-1     # after Ctrl-C, SIGTERM or a crash
cd .. && npm run gauntlet -- runs/ppo-1/best.json
```

`--help` lists the settings (stage lengths, network, learning rate and the rest of the PPO's, the Glutton reward's weights); they and how to run it in the background are in `AGENTS.md`.

## Other commands

`npm run bench -- --matches 500` plays many matches and prints ticks per second. `npm test` runs
the tests.
