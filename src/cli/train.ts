import { appendFileSync, existsSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { parseArgs } from "node:util";
import { Activation, Personality } from "snake-colyseus/bots";
import { DEFAULT_SETTINGS, TrainSettings, checkSettings, train } from "../train.ts";
import { exit, orExit, parseNumber } from "./args.ts";

const defaults = DEFAULT_SETTINGS;
const usage = `Trains a population of brains by neuroevolution, alone on the board and then against the rookie.
Writes settings.json, log.jsonl (one line a generation, printed as it goes) and best.json (the last
generation's fittest brain) into the run folder.

Usage: npm run train -- --personality glutton --run <folder> [options]

  --personality <p>      what to train for: only glutton for now
  --run <folder>         the run folder to write, which mustn't exist yet (runs/ is ignored by git)
  --seed <n>             the run's seed, a whole number in [0, 2^32) (default 1)
  --generations <n>      generations in all (default ${defaults.generations})
  --alone <n>            of them, how many are played alone before the rookie (default ${defaults.aloneGenerations})
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
  --tick-bonus <w>       fitness per tick survived (default ${defaults.tickBonus})`;

const options = {
  personality: { type: "string" },
  run: { type: "string" },
  seed: { type: "string", default: "1" },
  generations: { type: "string", default: String(defaults.generations) },
  alone: { type: "string", default: String(defaults.aloneGenerations) },
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
  help: { type: "boolean" }
} as const;

const args = orExit(usage, () => parseArgs({ options, strict: true }).values);
if (args.help) exit(usage, 0);

const { settings, run } = orExit(usage, () => {
  if (!args.personality) throw new Error("give a --personality");
  if (!args.run) throw new Error("give a --run folder");
  if (existsSync(args.run)) throw new Error(`${args.run} already exists: name a new run folder`);
  const number = (name: keyof typeof options) => parseNumber(name, args[name] as string);
  const settings: TrainSettings = {
    personality: args.personality as Personality,
    seed: number("seed"),
    generations: number("generations"),
    aloneGenerations: number("alone"),
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
    tickBonus: number("tick-bonus")
  };
  checkSettings(settings);
  return { settings, run: args.run };
});

mkdirSync(run, { recursive: true });
writeFileSync(join(run, "settings.json"), `${JSON.stringify(settings, null, 2)}\n`);
const { best } = train(settings, (line) => {
  const text = JSON.stringify(line);
  console.log(text);
  appendFileSync(join(run, "log.jsonl"), `${text}\n`);
});
writeFileSync(join(run, "best.json"), `${JSON.stringify(best, null, 2)}\n`);
console.log(`best brain: ${join(run, "best.json")}`);
