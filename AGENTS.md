# AGENTS.md

`snake-lab` trains and evaluates the snake game's AI opponents. It runs seeded headless games far
faster than real time, runs the evaluation gauntlet that decides whether a snake joins the roster,
and holds the trainers (neuroevolution in TypeScript, then PPO in Python), the training dashboard
and the experiment logs.

**Status:** plays seeded headless matches (slice 09, #2). The gauntlet is next (#1); no trainers yet.

## Stack and layout

TypeScript on Node (>= 20.9), run with `tsx`; no build step. Tests use `node:test`, and the CLIs parse
with `node:util`'s `parseArgs`. Keep dependencies few.

- `src/match.ts`: `playMatch`, one round between 1 to 4 bots from a seed. Everything the rest of
  the lab plays through. Its loop follows `SnakeRoom`'s order exactly (record every bot's
  snapshots, steer every live bot, run the engine's tick), so change it only when the room changes.
- `src/cli/`: the `match` and `bench` commands, and the options they share.
- `test/`: `*.test.ts`, one per module.

## Commands

- `npm install`: installs the engine and bots from `snake-colyseus` at the pinned tag (its
  `prepare` builds them). If `NODE_ENV=production` is set, add `--include=dev`.
- `npm test`, `npm run typecheck`
- `npm run match -- --seed 7 --players rookie,dummy --mode timed --fps 8 --delays 2`: one match,
  printed. `--events` lists every event; `--json` prints the whole result. `--help` for the rest.
- `npm run bench -- --matches 500 --players rookie,rookie`: plays seeds one after another and
  prints ticks per second. Run it before and after a change that could slow matches down.

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
