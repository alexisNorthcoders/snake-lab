# AGENTS.md

`snake-lab` trains and evaluates the snake game's AI opponents. It runs seeded headless games far
faster than real time, runs the evaluation gauntlet that decides whether a snake joins the roster,
and holds the trainers (neuroevolution in TypeScript, then PPO in Python), the training dashboard
and the experiment logs.

**Status:** plays seeded headless matches (slice 09, #2), runs the gauntlet (#3) and trains a Glutton
by neuroevolution (#7) on every core, with a checkpoint every generation and resuming (#8), promotes a generation to the roster (#9), and trains all three personalities through the league (#10). No PPO yet.

## Stack and layout

TypeScript on Node (>= 20.9), run with `tsx`; no build step. Tests use `node:test`, and the CLIs parse
with `node:util`'s `parseArgs`. Keep dependencies few.

- `src/match.ts`: `playMatch`, one round between 1 to 4 bots from a seed. Everything the rest of
  the lab plays through. Its loop follows `SnakeRoom`'s order exactly (record every bot's
  snapshots, steer every live bot, run the engine's tick), so change it only when the room changes.
- `src/gauntlet.ts`: the gauntlet. `loadCandidate` (roster id or brain file), `opponentsFor`,
  `runGauntlet` (the grid of opponent × delay 0-4 × mode, N seeds each played in both seats) and
  `verdict` (the bar: at least 60% of matches won against the rookie at delay 2, both modes; draws
  count as not winning). `promote` uses it. `formatReport` prints a report.
- `src/train.ts`: the neuroevolution trainer. `train` (the whole run, or the rest of one from a
  `Checkpoint`), and its parts: `randomBrain`, `crossover`, `mutate`, `breed`, `fixtures` (a
  generation's matches), `evaluate`, `survivorFitness`, `hunterFitness`, `kills`, `gluttonFitness`. A genome is a brain in `snake-colyseus/bots`'
  format, played through `brainDecider`, so it's saved with no conversion.
- `src/pool.ts` and `src/worker.ts`: the worker threads a generation's snakes are shared out to, one
  snake a job, each fitness put back in the snake's place.
- `src/run.ts`: the run folder. `createRun`, `trainRun` (from generation 0 or the latest
  checkpoint), `saveGeneration` (write then rename, the checkpoint last) and `runFiles`.
- `src/promote.ts`: `promote` (a generation through the gauntlet, then into a `snake-colyseus` checkout), `slug`, `bumpMinor`.
- `src/dashboard/`: the training dashboard. `replay.ts` (the grid's games: `sampleFixtures`, `gameStream`, `readGenerationBest`), `network.ts` (`networkLayout`: a brain's columns, labels and edges), `log.ts` (`LogReader`: reads a run's `log.jsonl`, whole lines only, and reports an append or, after a resume rewrote lines, a reset), `server.ts` (`startDashboard`: Node's http server, the page, `/api/run`, `/events` and `/games` as Server-Sent Events) and `page.html` (plain HTML, CSS and JS, the chart hand-drawn in SVG). It only reads the run folder.
- `src/cli/`: the `match`, `bench`, `gauntlet`, `train`, `promote` and `dashboard` commands, and the options they share.
- `docs/benchmarks/`: gauntlet and training speeds worth keeping, each with its command, engine tag, machine and date.
- `test/`: `*.test.ts`, one per module.

## Commands

- `npm install`: installs the engine and bots from `snake-colyseus` at the pinned tag (its
  `prepare` builds them). If `NODE_ENV=production` is set, add `--include=dev`.
- `npm test`, `npm run typecheck`
- `npm run match -- --seed 7 --players rookie,dummy --mode timed --fps 8 --delays 2`: one match,
  printed. `--players` takes roster ids or brain files. `--events` lists every event; `--json` prints the whole result. `--help` for the rest.
- `npm run bench -- --matches 500 --players rookie,rookie`: plays seeds one after another and
  prints ticks per second. Run it before and after a change that could slow matches down.
- `npm run gauntlet -- rookie` (or a path to a brain file): plays the candidate against the rookie
  and every other roster snake, prints win, loss and draw rates per opponent, delay and mode, and
  the verdict. Exits 0 on a pass and 1 on a fail. `--seeds` (default 100), `--base-seed` (default 1),
  `--fps` (default 8). On the Pi it plays about 14 matches a second: about 5 minutes for the rookie
  at the defaults.
- `npm run train -- --personality glutton --run runs/glutton-1 --seed 1`: trains by neuroevolution
  and writes the run folder (see "Long runs"), printing each generation's log line as it goes and a
  summary at the end: matches a second and generations an hour. The folder mustn't exist yet;
  `runs/` is ignored by git. `--personality` is `glutton`, `survivor` or `hunter` (see "Personalities and stages"). `--help` lists
  every setting. `--workers` (default: one a core) sets the threads. Any saved brain plays in
  `match --players` and the `gauntlet`.
- `npm run train -- --resume runs/glutton-1`: carries on from the latest checkpoint. Only
  `--generations` (to train further, or finish) and `--workers` may be given.

- `npm run promote -- runs/glutton-1 50 --name "Nimble Pete"`: puts that generation's brain through
  the gauntlet (`--seeds`, `--base-seed`, `--fps` as there). On a pass it writes `brains/<id>.json`
  and `entries/<id>.json` under `src/bots/` in the `snake-colyseus` checkout (`--target`, default
  `../snake-colyseus`), bumps its `engineVersion` by a minor version, prints a PR description and
  saves it as `promote-<id>.md` in the run folder. On a fail it prints the table and the verdict,
  exits 1 and writes nothing. It also refuses, writing nothing, when the id (a slug of the name) or
  the name is in the target's roster, the target isn't a `snake-colyseus` checkout, or the
  generation's brain is missing or invalid. It never runs git. See "Promoting".

- `npm run dashboard -- runs/glutton-1 [--port 8080] [--host 127.0.0.1] [--speed 8] [--games 6]`: serves a page for one run: its
  settings, the latest generation and stage, whether the log is still being written to, and a chart of best
  and mean fitness per generation with the stages shaded. It grows as the run trains, with no reload (see "The dashboard").

## The dashboard

It reads `settings.json` and `log.jsonl` from the run folder and never writes there; the trainer publishes
nothing. The page gets the whole log when it connects and then one `generation` event per new line (a `log`
event with the whole log again if a resume rewrote lines). If the connection drops, the browser reconnects and
is sent the whole log. A folder without a valid `settings.json` is refused. One run per server.

**The grid of sample games.** Under the chart, a grid shows a generation's best snake (`generations/gen-NNNN.json`)
playing that generation's own fixtures, replayed from the seed through `drawGeneration`, never recorded. Fixtures
against a population opponent are left out (that population isn't on disk); alone, rookie and roster-opponent
fixtures are kept. `GET /games?generation=N` plays them on the server and streams SSE: `generation`, then each game's
`start` (opponents, mode, delays, seat), a `tick` per board (0 is the board as dealt: cells, food, scores, deaths) and an
`end` (the result), then `done`; 404 if that generation isn't saved. The page draws each game on a canvas at `--speed`
ticks a second (default 8, editable on the page), loops it, and labels it with opponent, mode, delays and result. It
follows the latest generation; click the chart to pin one, "follow the latest" to go back. `--games` (default 6) caps
the games shown per generation, to keep the replays light next to the trainer's workers.

**The network visualiser.** Beside the grid the page draws the network of the snake in the followed game (the first
by default; click a game to follow another). The `generation` event carries `networkLayout(brain)`: a column of
labelled nodes a layer (encoder v1's `INPUT_LABELS`, hidden units numbered, `OUTPUT_LABELS`) and an edge a weight, for
any hidden sizes. Each `tick` message carries `activations`: `values`, every layer's values from the package's
`layerValues` for the snake's encoded view when it chose the move that made that tick (the lab copies neither the forward
pass nor the encoder), and `chosen`, the output (0 left, 1 straight, 2 right) of the move it made. Tick 0 has none. The page
shades nodes by value, marks the winning output, and redraws the network when the generation's brain changes. It needs
`snake-colyseus` at `engine-v4.1.0` or later.

Run it on the same box as the run. It listens on `127.0.0.1`, so from your own machine:

```sh
ssh -L 8080:127.0.0.1:8080 pi@<the Pi>      # then open http://localhost:8080
```

`--host 0.0.0.0` (or the Pi's LAN address) listens on the LAN: only do that on the Pi's home network. It has no
login: never expose it publicly on the VPS, keep it on `127.0.0.1` and use `ssh -L`. Run it in the background
like a trainer (see "In the background"), directly with `npx tsx src/cli/dashboard.ts <run>`.

## Promoting

`promote` only writes data, and a person reviews every promotion. After a pass:

1. In the `snake-colyseus` checkout, branch, commit the three changed files (the brain, the entry,
   `package.json`) and open the PR with the printed description (or `promote-<id>.md`).
2. Once it's merged and CI has tagged the new `engine-vX.Y.Z`, bump the pin in this repo's
   `package.json` to it and `npm install`.

## The trainer's settings

Every one is a `train` option (`--help`) and a field of `TrainSettings`; defaults in `DEFAULT_SETTINGS`.

| Setting | Default | What it does |
| --- | --- | --- |
| `generations` | 50 | generations in all |
| `aloneGenerations` (`--alone`) | 10 | the first ones, played alone on the board |
| `rookieGenerations` (`--rookie`) | 20 | the next ones, 1v1 against the rookie; the rest, to the end, are the league |
| `fourPlayerShare` (`--four-player`) | 0.2 | the league's share of four-player matches; the rest are 1v1 |
| `population` | 50 | snakes a generation |
| `elites` | 2 | the fittest, copied into the next generation unchanged |
| `tournament` | 3 | a parent is the fittest of this many drawn at random |
| `mutationRate` | 0.1 | each weight and bias's chance of a nudge in a child |
| `mutationSize` | 0.2 | a nudge's standard deviation (Gaussian) |
| `initialSize` | 0.5 | the first generation's weights' standard deviation |
| `hidden` | `[16]` | hidden layer sizes; in is encoder v1 (23), out is left, straight, right |
| `activation` | `tanh` | the hidden layers' activation |
| `matches` | 10 | matches each snake plays a generation: every snake the same ones |
| `fps` | 8 | ticks per second, which sets a timed round's length |
| `foodWeight`, `tickBonus` | 1, 0.1 | Glutton fitness: food score × `foodWeight` + ticks survived × `tickBonus`, averaged over the snake's matches |
| `aliveWeight`, `survivorWinBonus` | 1, 200 | Survivor fitness: ticks alive × `aliveWeight` + `survivorWinBonus` for a win |
| `killBonus`, `hunterWinBonus`, `foodBonus` | 100, 50, 0.1 | Hunter fitness: kills × `killBonus` + `hunterWinBonus` for a win + food score × `foodBonus` |

A child's parents cross unit by unit: each unit takes its incoming weights and bias whole from one
parent. A generation's matches draw their seed, mode (both), opponents, seat and each seat's delay (0 to 4,
separately) from the generation's generator. At the defaults a generation is 500 matches: on the Pi's
4 cores, about 6 seconds a generation while they play alone and longer as they learn to live
against the rookie (see `docs/benchmarks/`).

## Personalities and stages

A win is the engine's `winnerId`. A kill is a `died` event whose `by` is the snake, except a head-on
collision (`cause: "head-on"`): the engine names each snake the other as `by`, but both die, so it
counts as a kill for neither.

Generations go through three **stages**, named in each log line and in the checkpoint (which holds
the stage of the generation it carries on with, so a resume lands in the right one):

- `alone`: the first `aloneGenerations`, one snake on the board.
- `rookie`: the next `rookieGenerations`, 1v1 against the rookie.
- `league`: the rest, to the end. Each match's opponents are drawn from the generation's seed, half
  from the current population (as it was played) and half from the roster in the pinned package,
  the rookie included. `fourPlayerShare` of the matches have three opponents; the rest have one.
  Every snake in a generation plays the same fixtures, so faces the same opponents.

A run from before the league has no `rookieGenerations`: it stays in the rookie stage to its end.

## Long runs

A run folder holds:

- `settings.json`: the run's settings. Resuming only ever changes `generations` in it.
- `log.jsonl`: one `{generation, stage, best, mean}` line a generation.
- `generations/gen-NNNN.json`: every generation's fittest, a brain file, so any generation can be
  promoted later (a difficulty ladder: gen 10, 50, 500). `best.json` is the latest one again.
- `checkpoint.json`: the next generation's number and its whole population, bred from the last one
  played. It's the only population on disk, overwritten each generation.

Each file is written to `<name>.tmp` and renamed into place, the checkpoint last of a generation's
files, so a crash mid-write leaves the previous checkpoint whole. A resume drops log lines past the
checkpoint and replays from it, overwriting that generation's brain file. Because every draw in
generation `g` comes from `generationRng(seed, g)`, a resumed run's logs and brains are byte for byte
those of a run that never stopped, and `--generations` can raise a finished run's target and carry on.

**Stopping.** Ctrl-C or SIGTERM lets the current generation finish and be saved, then exits 0 with
a `Stopped:` line. A second one abandons the generation (exit 130); SIGKILL or a crash does the same.
Either way the latest checkpoint is intact: `--resume` it.

**Workers.** A generation's snakes are shared out to worker threads (`src/pool.ts`), by default one
a core (`os.availableParallelism()`); `--workers 1` plays on the main thread. Fitnesses go back in
the population's order, so the number of workers never changes a result, and it can differ between
a run and its resume. A snake that throws, or a worker that dies, fails the run loudly (exit 1,
checkpoint kept); it's never scored as zero. Leave a core free with `--workers 3` if the machine
has other work.

**In the background.** Train on the Pi or the VPS, never on the explorer machine. Work out the
length first: at the defaults the Pi's 4 cores train about 550 generations an hour while the snakes
still die young, far fewer once they live long (see `docs/benchmarks/`). Any of these survives the
shell closing:

Run `tsx` directly rather than through `npm run`: npm doesn't pass a signal on to the trainer, so
stopping npm would leave the trainer running without it. `tsx` does pass signals on.

- `nohup npx tsx src/cli/train.ts --personality glutton --run runs/glutton-1 --seed 1 > runs/glutton-1.out 2>&1 &`,
  then `tail -f runs/glutton-1.out`. Stop it with `pkill -INT -f "train.ts.*runs/glutton-1"`.
- `tmux new -s train`, run `npm run train -- ...` there, detach with Ctrl-B D, come back with
  `tmux attach -t train`, and stop it with Ctrl-C.
- pm2: `pm2 start node_modules/.bin/tsx --interpreter none --name glutton-1 --no-autorestart --kill-timeout 600000 -- src/cli/train.ts --personality glutton --run runs/glutton-1 --seed 1`,
  logs with `pm2 logs glutton-1`, and stop with `pm2 stop glutton-1` (the kill timeout gives the
  generation time to finish). Carry on with the same line but `-- src/cli/train.ts --resume runs/glutton-1`
  after `pm2 delete glutton-1`. Never restart, stop or delete the `agent-runner` process.

After a stop or crash, `npm run train -- --resume runs/glutton-1` carries on.

## Determinism

A match is a pure function of its options: the same seed, seats (players and delays), mode and fps
always give the same result and events. Everything random draws from the engine's `mulberry32`,
seeded once per match; nothing in a match reads the clock or `Math.random`. Keep it that way:
the gauntlet and trainers rely on replaying a match from its seed. The trainer draws everything in
generation `g` (the first brains, the matches, the breeding) from `generationRng(seed, g)`, so the
same seed and settings give the same log and brains, whatever the workers and wherever it was
stopped and resumed.

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
