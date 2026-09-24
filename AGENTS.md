# AGENTS.md

`snake-lab` trains and evaluates the snake game's AI opponents. It runs seeded headless games far
faster than real time, runs the evaluation gauntlet that decides whether a snake joins the roster,
and holds the trainers (neuroevolution in TypeScript, then PPO in Python), the training dashboard
and the experiment logs.

**Status:** empty. The first code arrives with slice 09 of the snake ML plan (headless matches
and the gauntlet), which waits on the engine being installable from `snake-colyseus`.

## How it relates to the game

- **The rules are never copied.** The engine (and later the encoders and the network forward
  pass) comes from `snake-colyseus` as a git dependency pinned to an `engine-vX.Y.Z` tag, imported
  from `snake-colyseus/engine`. A rules change happens in `snake-colyseus` and arrives here by
  bumping the tag.
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
