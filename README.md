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
and `--json` prints the whole result. Players are roster ids (`rookie`, `dummy`), seated in the order
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

## Other commands

`npm run bench -- --matches 500` plays many matches and prints ticks per second. `npm test` runs
the tests.
