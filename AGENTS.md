# AGENTS.md

`snake-lab` trains and evaluates the snake game's AI opponents. It runs seeded headless games far
faster than real time, runs the evaluation gauntlet that decides whether a snake joins the roster,
and holds the trainers (neuroevolution in TypeScript, then PPO in Python), the training dashboard
and the experiment logs.

**Status:** plays seeded headless matches (slice 09, #2) and runs the gauntlet (#3). No trainers yet.

## Stack and layout

TypeScript on Node (>= 20.9), run with `tsx`; no build step. Tests use `node:test`, and the CLIs parse
with `node:util`'s `parseArgs`. Keep dependencies few.

- `src/match.ts`: `playMatch`, one round between 1 to 4 bots from a seed. Everything the rest of
  the lab plays through. Its loop follows `SnakeRoom`'s order exactly (record every bot's
  snapshots, steer every live bot, run the engine's tick), so change it only when the room changes.
- `src/gauntlet.ts`: the gauntlet. `loadCandidate` (roster id or brain file), `opponentsFor`,
  `runGauntlet` (the grid of opponent × delay 0-4 × mode, N seeds each played in both seats) and
  `verdict` (the bar: at least 60% of matches won against the rookie at delay 2, both modes; draws
  count as not winning). `promote` will use it.
- `src/cli/`: the `match`, `bench` and `gauntlet` commands, and the options they share.
- `docs/benchmarks/`: gauntlet results worth keeping, each with its command, engine tag and date.
- `test/`: `*.test.ts`, one per module.

## Commands

- `npm install`: installs the engine and bots from `snake-colyseus` at the pinned tag (its
  `prepare` builds them). If `NODE_ENV=production` is set, add `--include=dev`.
- `npm test`, `npm run typecheck`
- `npm run match -- --seed 7 --players rookie,dummy --mode timed --fps 8 --delays 2`: one match,
  printed. `--events` lists every event; `--json` prints the whole result. `--help` for the rest.
- `npm run bench -- --matches 500 --players rookie,rookie`: plays seeds one after another and
  prints ticks per second. Run it before and after a change that could slow matches down.
- `npm run gauntlet -- rookie` (or a path to a brain file): plays the candidate against the rookie
  and every other roster snake, prints win, loss and draw rates per opponent, delay and mode, and
  the verdict. Exits 0 on a pass and 1 on a fail. `--seeds` (default 100), `--base-seed` (default 1),
  `--fps` (default 8). On the Pi it plays about 14 matches a second: about 5 minutes for the rookie
  at the defaults.

## Determinism

A match is a pure function of its options: the same seed, seats (players and delays), mode and fps
always give the same result and events. Everything random draws from the engine's `mulberry32`,
seeded once per match; nothing in a match reads the clock or `Math.random`. Keep it that way:
the gauntlet and trainers rely on replaying a match from its seed.

## How it relates to the game

- **The rules are never copied.** The engine, and the bots' view, reaction-delay snapshots,
  encoders, brains and roster, come from `snake-colyseus` as a git dependency pinned to an `engine-vX.Y.Z` tag, imported
  from `snake-colyseus/engine` and `snake-colyseus/bots` (never from their `build/` files). A rules
  change happens in `snake-colyseus` and arrives here by bumping the tag in `package.json`.
- **Only brain files flow back.** Promoting a snake writes its brain and roster entry into
  `snake-colyseus`, through a PR there that a human reviews.
- This repo makes no calls to the game's servers.

## Agent skills

### Issue tracker

Issues live in GitHub Issues for `alexisNorthcoders/snake-lab`, managed with the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Uses the five default triage labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
