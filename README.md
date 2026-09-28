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

`npm run bench -- --matches 500` plays many matches and prints ticks per second. `npm test` runs
the tests.
