import { availableParallelism } from "node:os";
import { performance } from "node:perf_hooks";
import { parseArgs } from "node:util";
import { type Activation, type Personality } from "snake-colyseus/bots";
import { DEFAULT_SETTINGS, type TrainSettings } from "../train.ts";
import { createRun, openRun, runFiles, trainRun } from "../run.ts";
import { exit, orExit, parseNumber } from "./args.ts";

const defaults = DEFAULT_SETTINGS;
const usage = `Trains a population of brains by neuroevolution, alone on the board, then against the rookie, then in the league.
Writes into the run folder: settings.json, log.jsonl (one line a generation, printed as it goes),
generations/gen-NNNN.json (each generation's fittest brain), best.json (the latest one's) and
checkpoint.json (the population to carry on from). Ctrl-C or SIGTERM stops after the current
generation; a second one abandons it. Either way the latest checkpoint is kept.

Usage: npm run train -- --personality glutton --run <folder> [options]
       npm run train -- --resume <folder> [--generations <n>] [--workers <n>]

  --personality <p>      what to train for: glutton, survivor or hunter
  --run <folder>         the run folder to write, which mustn't exist yet (runs/ is ignored by git)
  --resume <folder>      carry on a run from its latest checkpoint, with its settings
  --workers <n>          threads to play matches on; never changes the result (default ${availableParallelism()}, one a core)
  --seed <n>             the run's seed, a whole number in [0, 2^32) (default 1)
  --generations <n>      generations in all (default ${defaults.generations})
  --alone <n>            of them, how many are played alone first (default ${defaults.aloneGenerations})
  --rookie <n>           then how many against the rookie; the rest are the league (default ${defaults.rookieGenerations})
  --four-player <p>      the league's share of four-player matches (default ${defaults.fourPlayerShare})
  --population <n>       snakes a generation (default ${defaults.population})
  --elites <n>           the fittest, kept unchanged (default ${defaults.elites})
  --tournament <n>       snakes a parent is the fittest of (default ${defaults.tournament})
  --mutation-rate <p>    each weight's chance of a nudge (default ${defaults.mutationRate})
  --mutation-size <s>    a nudge's standard deviation (default ${defaults.mutationSize})
  --initial-size <s>     the first generation's weights' standard deviation (default ${defaults.initialSize})
  --hidden <n,...>       hidden layer sizes (default ${defaults.hidden.join(",")})
  --activation <a>       tanh, relu or sigmoid (default ${defaults.activation})
  --matches <n>          matches each snake plays a generation (default ${defaults.matches})
  --fps <n>              ticks per second, which sets a timed round's length (default ${defaults.fps})
  --food-weight <w>      fitness per point of food score (default ${defaults.foodWeight})
  --tick-bonus <w>       glutton fitness per tick survived (default ${defaults.tickBonus})
  --glutton-win-bonus <w>  glutton fitness for a win (default ${defaults.gluttonWinBonus})
  --alive-weight <w>     survivor fitness per tick alive (default ${defaults.aliveWeight})
  --survivor-win-bonus <w>  survivor fitness for a win (default ${defaults.survivorWinBonus})
  --kill-bonus <w>       hunter fitness per kill (default ${defaults.killBonus})
  --hunter-win-bonus <w> hunter fitness for a win (default ${defaults.hunterWinBonus})
  --food-bonus <w>       hunter fitness per point of food score (default ${defaults.foodBonus})`;

const options = {
  personality: { type: "string" },
  run: { type: "string" },
  seed: { type: "string", default: "1" },
  generations: { type: "string", default: String(defaults.generations) },
  alone: { type: "string", default: String(defaults.aloneGenerations) },
  rookie: { type: "string", default: String(defaults.rookieGenerations) },
  "four-player": { type: "string", default: String(defaults.fourPlayerShare) },
  population: { type: "string", default: String(defaults.population) },
  elites: { type: "string", default: String(defaults.elites) },
  tournament: { type: "string", default: String(defaults.tournament) },
  "mutation-rate": { type: "string", default: String(defaults.mutationRate) },
  "mutation-size": { type: "string", default: String(defaults.mutationSize) },
  "initial-size": { type: "string", default: String(defaults.initialSize) },
  hidden: { type: "string", default: defaults.hidden.join(",") },
  activation: { type: "string", default: defaults.activation },
  matches: { type: "string", default: String(defaults.matches) },
  fps: { type: "string", default: String(defaults.fps) },
  "food-weight": { type: "string", default: String(defaults.foodWeight) },
  "tick-bonus": { type: "string", default: String(defaults.tickBonus) },
  "glutton-win-bonus": { type: "string", default: String(defaults.gluttonWinBonus) },
  "alive-weight": { type: "string", default: String(defaults.aliveWeight) },
  "survivor-win-bonus": { type: "string", default: String(defaults.survivorWinBonus) },
  "kill-bonus": { type: "string", default: String(defaults.killBonus) },
  "hunter-win-bonus": { type: "string", default: String(defaults.hunterWinBonus) },
  "food-bonus": { type: "string", default: String(defaults.foodBonus) },
  resume: { type: "string" },
  workers: { type: "string", default: String(availableParallelism()) },
  help: { type: "boolean" }
} as const;

const { values: args, tokens } = orExit(usage, () => parseArgs({ options, strict: true, tokens: true }));
if (args.help) exit(usage, 0);
/** The options given on the command line, not left at their defaults. */
const given = new Set(tokens.flatMap((token) => (token.kind === "option" ? [token.name] : [])));
const number = (name: keyof typeof options) => parseNumber(name, args[name] as string);

const { run, workers, generations } = orExit(usage, () => {
  const workers = number("workers");
  if (!Number.isInteger(workers) || workers < 1) throw new Error(`--workers must be a whole number of at least 1, not ${workers}`);
  if (args.resume !== undefined) {
    const fixed = [...given].filter((name) => !["resume", "generations", "workers"].includes(name));
    if (fixed.length > 0) {
      throw new Error(`a resumed run keeps its settings: only --generations and --workers can be given, not ${fixed.map((n) => `--${n}`).join(", ")}`);
    }
    openRun(args.resume);
    return { run: args.resume, workers, generations: given.has("generations") ? number("generations") : undefined };
  }
  if (!args.personality) throw new Error("give a --personality");
  if (!args.run) throw new Error("give a --run folder");
  const settings: TrainSettings = {
    personality: args.personality as Personality,
    seed: number("seed"),
    generations: number("generations"),
    aloneGenerations: number("alone"),
    rookieGenerations: number("rookie"),
    fourPlayerShare: number("four-player"),
    population: number("population"),
    elites: number("elites"),
    tournament: number("tournament"),
    mutationRate: number("mutation-rate"),
    mutationSize: number("mutation-size"),
    initialSize: number("initial-size"),
    hidden: args.hidden.split(",").map((size) => parseNumber("hidden", size)),
    activation: args.activation as Activation,
    matches: number("matches"),
    fps: number("fps"),
    foodWeight: number("food-weight"),
    tickBonus: number("tick-bonus"),
    gluttonWinBonus: number("glutton-win-bonus"),
    aliveWeight: number("alive-weight"),
    survivorWinBonus: number("survivor-win-bonus"),
    killBonus: number("kill-bonus"),
    hunterWinBonus: number("hunter-win-bonus"),
    foodBonus: number("food-bonus")
  };
  createRun(args.run, settings);
  return { run: args.run, workers, generations: undefined };
});

let stopping = false;
const onSignal = (signal: NodeJS.Signals) => {
  if (stopping) exit(`${signal} again: abandoning the generation. The latest checkpoint is kept.`, 130);
  stopping = true;
  console.error(`${signal}: stopping after this generation (again to abandon it)`);
};
process.on("SIGINT", onSignal);
process.on("SIGTERM", onSignal);

const { settings } = openRun(run);
const target = generations ?? settings.generations;
const start = performance.now();
const result = await trainRun(run, {
  workers,
  generations,
  stop: () => stopping,
  onGeneration: (line) => console.log(JSON.stringify(line))
}).catch((error: Error) => exit(`training failed, the latest checkpoint is kept: ${error.stack ?? error.message}`));
if (!result) exit(`${run} already has all its ${target} generations: give more with --generations`, 0);

const seconds = (performance.now() - start) / 1000;
const { log } = result;
const last = log.at(-1)!.generation;
console.log(`${last < target - 1 ? "Stopped" : "Trained"}: generations ${log[0].generation} to ${last} of ${target} ` +
  `in ${seconds.toFixed(1)}s on ${workers} worker${workers === 1 ? "" : "s"}: ` +
  `${((log.length * settings.population * settings.matches) / seconds).toFixed(1)} matches a second, ` +
  `${Math.round((log.length * 3600) / seconds)} generations an hour`);
console.log(`best brain: ${runFiles(run).best}`);
