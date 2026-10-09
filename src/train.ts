import { type GameMode, RULES_VERSION, type Rng, mulberry32 } from "snake-colyseus/engine";
import {
  type Activation,
  BRAIN_FORMAT,
  BRAIN_FORMAT_VERSION,
  type Brain,
  ENCODER_SIZE,
  ENCODER_VERSION,
  type Personality,
  brainDecider,
  pickBot,
  roster
} from "snake-colyseus/bots";
import { type MatchOptions, type MatchResult, type Player, playMatch } from "./match.ts";
import { DELAYS, MODES } from "./gauntlet.ts";
import { Pool } from "./pool.ts";

/** The personalities the trainer has a fitness for. */
export const TRAINABLE: readonly Personality[] = ["glutton", "survivor", "hunter"];

/** Alone on the board, then 1v1 against the rookie, then the league: varied opponents, some matches four-player. */
export type Stage = "alone" | "rookie" | "league";

/** Everything that decides a run: the same settings always train the same brains. */
export interface TrainSettings {
  personality: Personality;
  /** Every random draw in the run derives from this and the generation: a whole number in [0, 2^32). */
  seed: number;
  /** How many generations in all. */
  generations: number;
  /** How many of them, from the first, are played alone. */
  aloneGenerations: number;
  /** How many come after those, against the rookie; the rest, to the end, are the league. */
  rookieGenerations: number;
  /** In the league, the share of matches that are four-player, in [0, 1]; the rest are 1v1. */
  fourPlayerShare: number;
  /** Snakes in each generation. */
  population: number;
  /** The best few, copied into the next generation unchanged. */
  elites: number;
  /** How many snakes a parent is picked from: the fittest of that many, drawn at random. */
  tournament: number;
  /** The chance each weight and bias is nudged in a child. */
  mutationRate: number;
  /** The standard deviation of a nudge. */
  mutationSize: number;
  /** The standard deviation of the first generation's weights and biases. */
  initialSize: number;
  /** The hidden layers' sizes, between the encoder's inputs and the three outputs. */
  hidden: number[];
  activation: Activation;
  /** Matches each snake plays a generation, all snakes the same ones. */
  matches: number;
  fps: number;
  /** Glutton fitness: `foodWeight` × score + `tickBonus` × ticks survived, averaged over its matches. */
  foodWeight: number;
  tickBonus: number;
  /** Survivor fitness: `aliveWeight` × ticks alive + `survivorWinBonus` if it won, averaged over its matches. */
  aliveWeight: number;
  survivorWinBonus: number;
  /** Hunter fitness: `killBonus` × kills + `hunterWinBonus` if it won + `foodBonus` × food score, averaged over its matches. */
  killBonus: number;
  hunterWinBonus: number;
  foodBonus: number;
}

export const DEFAULT_SETTINGS: Omit<TrainSettings, "personality" | "seed"> = {
  generations: 50,
  aloneGenerations: 10,
  rookieGenerations: 20,
  fourPlayerShare: 0.2,
  population: 50,
  elites: 2,
  tournament: 3,
  mutationRate: 0.1,
  mutationSize: 0.2,
  initialSize: 0.5,
  hidden: [16],
  activation: "tanh",
  matches: 10,
  fps: 8,
  foodWeight: 1,
  tickBonus: 0.1,
  aliveWeight: 1,
  survivorWinBonus: 200,
  killBonus: 100,
  hunterWinBonus: 50,
  foodBonus: 0.1
};

/** One generation's line in the log. */
export interface GenerationLog {
  generation: number;
  stage: Stage;
  best: number;
  mean: number;
}

export interface TrainResult {
  log: GenerationLog[];
  /** The last generation's fittest. */
  best: Brain;
  /** The last generation, as it was played. */
  population: Brain[];
}

/** An opponent: a roster snake by id, or a snake of the generation being played by its place in the population. */
export type Opponent = { roster: string } | { population: number };

/**
 * One match every snake in a generation plays: the snake takes seat `seat`, the
 * `opponents` the other seats in order, and `delays` (one a seat, in seat order)
 * are their reaction delays.
 */
export interface Fixture {
  seed: number;
  mode: GameMode;
  seat: number;
  delays: number[];
  opponents: Opponent[];
}

/** Throws, saying why, if `settings` can't be trained. */
export function checkSettings(settings: TrainSettings) {
  const { personality, seed, generations, aloneGenerations, population, elites, tournament } = settings;
  if (!TRAINABLE.includes(personality)) {
    throw new Error(`personality must be ${TRAINABLE.join(", ")}, not ${personality}`);
  }
  const whole = (name: string, value: number, min: number) => {
    if (!Number.isInteger(value) || value < min) throw new Error(`${name} must be a whole number of at least ${min}, not ${value}`);
  };
  if (!Number.isInteger(seed) || seed < 0 || seed >= 2 ** 32) throw new Error(`seed must be a whole number in [0, 2^32), not ${seed}`);
  whole("generations", generations, 1);
  whole("aloneGenerations", aloneGenerations, 0);
  whole("rookieGenerations", settings.rookieGenerations, 0);
  if (aloneGenerations > generations) throw new Error(`aloneGenerations (${aloneGenerations}) can't be more than generations (${generations})`);
  whole("population", population, 2);
  whole("elites", elites, 0);
  if (elites >= population) throw new Error(`elites (${elites}) must be fewer than the population (${population})`);
  whole("tournament", tournament, 1);
  whole("matches", settings.matches, 1);
  settings.hidden.forEach((size) => whole("each hidden size", size, 1));
  if (!["tanh", "relu", "sigmoid"].includes(settings.activation)) throw new Error(`activation must be tanh, relu or sigmoid, not ${settings.activation}`);
  if (!(settings.mutationRate >= 0 && settings.mutationRate <= 1)) throw new Error(`mutationRate must be in [0, 1], not ${settings.mutationRate}`);
  if (!(settings.fourPlayerShare >= 0 && settings.fourPlayerShare <= 1)) throw new Error(`fourPlayerShare must be in [0, 1], not ${settings.fourPlayerShare}`);
  const weights = ["foodWeight", "tickBonus", "aliveWeight", "survivorWinBonus", "killBonus", "hunterWinBonus", "foodBonus"] as const;
  for (const name of ["mutationSize", "initialSize", "fps", ...weights] as const) {
    if (!Number.isFinite(settings[name]) || settings[name] < 0) throw new Error(`${name} must be a number of at least 0, not ${settings[name]}`);
  }
  if (settings.fps <= 0) throw new Error(`fps must be above 0, not ${settings.fps}`);
}

/** Generation `g`'s generator: seeded from the run's seed (scrambled, so no two runs share a stream) and `g` alone. */
export const generationRng = (seed: number, g: number): Rng =>
  mulberry32((Math.floor(mulberry32(seed)() * 2 ** 32) ^ Math.imul(g + 1, 0x9e3779b1)) >>> 0);

/** A standard normal draw (Box–Muller). */
export const gaussian = (rng: Rng) => Math.sqrt(-2 * Math.log(1 - rng())) * Math.cos(2 * Math.PI * rng());

const pick = <T>(rng: Rng, from: readonly T[]): T => from[Math.floor(rng() * from.length)];

/** A brain with every weight and bias drawn from a normal of standard deviation `size`. */
export function randomBrain(rng: Rng, hidden: number[], activation: Activation, size: number): Brain {
  const sizes = [ENCODER_SIZE, ...hidden, 3];
  const layers = sizes.slice(1).map((outputs, l) => ({
    weights: Array.from({ length: outputs }, () => Array.from({ length: sizes[l] }, () => gaussian(rng) * size)),
    biases: Array.from({ length: outputs }, () => gaussian(rng) * size)
  }));
  return {
    format: BRAIN_FORMAT,
    formatVersion: BRAIN_FORMAT_VERSION,
    encoderVersion: ENCODER_VERSION,
    rulesVersion: RULES_VERSION,
    sizes,
    activation,
    layers
  };
}

/** A child of two brains of the same shape: each unit takes its incoming weights and bias whole from one parent or the other. */
export function crossover(a: Brain, b: Brain, rng: Rng): Brain {
  return {
    ...a,
    sizes: [...a.sizes],
    layers: a.layers.map((layer, l) => {
      const units = layer.biases.map((_, j) => (rng() < 0.5 ? a : b).layers[l]);
      return {
        weights: units.map((from, j) => [...from.weights[j]]),
        biases: units.map((from, j) => from.biases[j])
      };
    })
  };
}

/** A copy of `brain` with each weight and bias, at chance `rate`, nudged by a normal of standard deviation `size`. */
export function mutate(brain: Brain, rng: Rng, rate: number, size: number): Brain {
  const nudge = (w: number) => (rng() < rate ? w + gaussian(rng) * size : w);
  return {
    ...brain,
    sizes: [...brain.sizes],
    layers: brain.layers.map(({ weights, biases }) => ({ weights: weights.map((row) => row.map(nudge)), biases: biases.map(nudge) }))
  };
}

/** The fittest of `size` snakes drawn at random (with replacement); the lower index wins a tie. */
const tournamentPick = (rng: Rng, fitness: number[], size: number) => {
  let best = Math.floor(rng() * fitness.length);
  for (let i = 1; i < size; i++) {
    const other = Math.floor(rng() * fitness.length);
    if (fitness[other] > fitness[best] || (fitness[other] === fitness[best] && other < best)) best = other;
  }
  return best;
};

/** Indices from fittest to least fit; the lower index first on a tie. */
export const ranked = (fitness: number[]) => fitness.map((_, i) => i).sort((i, j) => fitness[j] - fitness[i] || i - j);

/**
 * The next generation, the same size: the `elites` fittest copied unchanged,
 * then children of two tournament-picked parents, crossed and mutated.
 */
export function breed(population: Brain[], fitness: number[], rng: Rng, settings: Pick<TrainSettings, "elites" | "tournament" | "mutationRate" | "mutationSize">): Brain[] {
  const next = ranked(fitness).slice(0, settings.elites).map((i) => structuredClone(population[i]));
  while (next.length < population.length) {
    const a = population[tournamentPick(rng, fitness, settings.tournament)];
    const b = population[tournamentPick(rng, fitness, settings.tournament)];
    next.push(mutate(crossover(a, b, rng), rng, settings.mutationRate, settings.mutationSize));
  }
  return next;
}

/**
 * The generation's matches, all drawn from `rng`: seed, mode, then the
 * opponents (none alone, the rookie in its stage, in the league one, or three
 * in `fourPlayerShare` of the matches, each from the population or the roster
 * at even odds), the snake's seat, and each seat's delay (0 to 4).
 */
export const fixtures = (
  rng: Rng,
  stage: Stage,
  settings: Pick<TrainSettings, "matches" | "population" | "fourPlayerShare">,
  rosterIds: readonly string[] = roster.map((entry) => entry.id)
): Fixture[] =>
  Array.from({ length: settings.matches }, () => {
    const seed = Math.floor(rng() * 2 ** 32);
    const mode = pick(rng, MODES);
    const count = stage === "alone" ? 0 : stage === "rookie" ? 1 : rng() < settings.fourPlayerShare ? 3 : 1;
    const opponents: Opponent[] = Array.from({ length: count }, () =>
      stage === "rookie" ? { roster: "rookie" }
        : rng() < 0.5 ? { population: Math.floor(rng() * settings.population) } : { roster: pick(rng, rosterIds) });
    const seat = Math.floor(rng() * (count + 1));
    const delays = Array.from({ length: count + 1 }, () => pick(rng, DELAYS));
    return { seed, mode, seat, delays, opponents };
  });

/**
 * Everything generation `g` draws before breeding, from `generationRng(seed, g)`: generation 0's
 * random brains first, then the fixtures. The trainer and the dashboard both derive a generation
 * through this, so they can't drift apart. `rng` is left where breeding carries on.
 */
export function drawGeneration(settings: TrainSettings, g: number) {
  const rng = generationRng(settings.seed, g);
  const initial = g === 0
    ? Array.from({ length: settings.population }, () => randomBrain(rng, settings.hidden, settings.activation, settings.initialSize))
    : undefined;
  const stage = stageOf(g, settings);
  return { rng, initial, stage, fixtures: fixtures(rng, stage, settings) };
}

/** How many ticks the snake in `seat` lived: to the tick it died on, or the whole match. Needs the match's events. */
export const ticksSurvived = (match: MatchResult, seat: number) => {
  if (!match.events) throw new Error("ticks survived needs the match's events: play it with keepEvents");
  const death = match.events.find((e) => e.kind === "died" && e.player === `seat${seat}`);
  return death ? death.tick : match.ticks;
};

/** Glutton fitness: `foodWeight` × score plus `tickBonus` × ticks survived, averaged over the snake's matches. */
export function gluttonFitness(matches: { match: MatchResult; seat: number }[], weights: Pick<TrainSettings, "foodWeight" | "tickBonus">) {
  const total = matches.reduce((sum, { match, seat }) =>
    sum + weights.foodWeight * match.players[seat].score + weights.tickBonus * ticksSurvived(match, seat), 0);
  return total / matches.length;
}

/** Whether the snake in `seat` won the match. */
const won = (match: MatchResult, seat: number) => match.result.winnerId === `seat${seat}`;

/**
 * Kills by the snake in `seat`: `died` events whose `by` is it. A head-on
 * collision names each snake the other as `by`, but both die, so it counts for neither.
 */
export const kills = (match: MatchResult, seat: number) => {
  if (!match.events) throw new Error("kills need the match's events: play it with keepEvents");
  return match.events.filter((e) => e.kind === "died" && e.by === `seat${seat}` && e.cause !== "head-on").length;
};

type Played = { match: MatchResult; seat: number }[];

const average = (matches: Played, score: (match: MatchResult, seat: number) => number) =>
  matches.reduce((sum, { match, seat }) => sum + score(match, seat), 0) / matches.length;

/** Survivor fitness: `aliveWeight` × ticks alive plus `survivorWinBonus` for a win, averaged over the snake's matches. */
export const survivorFitness = (matches: Played, weights: Pick<TrainSettings, "aliveWeight" | "survivorWinBonus">) =>
  average(matches, (match, seat) => weights.aliveWeight * ticksSurvived(match, seat) + (won(match, seat) ? weights.survivorWinBonus : 0));

/** Hunter fitness: `killBonus` × kills plus `hunterWinBonus` for a win plus `foodBonus` × food score, averaged over the snake's matches. */
export const hunterFitness = (matches: Played, weights: Pick<TrainSettings, "killBonus" | "hunterWinBonus" | "foodBonus">) =>
  average(matches, (match, seat) => weights.killBonus * kills(match, seat) + (won(match, seat) ? weights.hunterWinBonus : 0) +
    weights.foodBonus * match.players[seat].score);

/** The stage generation `g` is played in. */
export const stageOf = (g: number, settings: Pick<TrainSettings, "aloneGenerations" | "rookieGenerations">): Stage =>
  g < settings.aloneGenerations ? "alone" : g < settings.aloneGenerations + settings.rookieGenerations ? "rookie" : "league";

/** The match `fixture` is, with `snake` in its seat; `population` is what its population opponents are picked from. */
export function fixtureOptions(fixture: Fixture, snake: Player, population: Brain[], fps: number): MatchOptions {
  const { seed, mode, seat, delays, opponents } = fixture;
  const others = opponents.map((opponent) =>
    "roster" in opponent ? pickBot(opponent.roster) : { name: `Opponent ${opponent.population}`, decider: brainDecider(population[opponent.population]) });
  const players = [...others.slice(0, seat), snake, ...others.slice(seat)];
  return { seed, mode, fps, seats: players.map((player, i) => ({ player, delay: delays[i] })) };
}

/** A snake's fitness over the generation's fixtures; `population` is what the fixtures' population opponents are picked from. */
export function evaluate(brain: Brain, fixtures: Fixture[], population: Brain[], settings: TrainSettings) {
  const snake = { name: "Candidate", decider: brainDecider(brain) };
  const played = fixtures.map((fixture) => ({ match: playMatch(fixtureOptions(fixture, snake, population, settings.fps)), seat: fixture.seat }));
  switch (settings.personality) {
    case "survivor": return survivorFitness(played, settings);
    case "hunter": return hunterFitness(played, settings);
    default: return gluttonFitness(played, settings);
  }
}

/** Where a run carries on from: the generation to play next, and its snakes. */
export interface Checkpoint {
  generation: number;
  /** The stage that generation is played in. */
  stage: Stage;
  population: Brain[];
}

export interface TrainOptions {
  /** Worker threads to play the matches on; 1 plays them on this thread. Never changes the result. */
  workers?: number;
  /** Carry on from here rather than from generation 0's random brains. */
  from?: Checkpoint;
  /** Called after each generation with its log line, its fittest, and the checkpoint to carry on from. */
  onGeneration?: (line: GenerationLog, best: Brain, next: Checkpoint) => void;
  /** Asked after each generation: true ends the run there, with the checkpoint saved. */
  stop?: () => boolean;
}

/**
 * Neuroevolution: trains a population of encoder-v1 brains for a personality.
 *
 * Each generation `g` draws everything random from `generationRng(seed, g)`:
 * the first generation's brains (`initialSize`), then the generation's
 * `matches` fixtures (seed, mode, opponents, seat and each seat's delay 0 to
 * 4), which every snake
 * plays so their fitness can be compared, then the next generation. That keeps
 * the `elites` fittest unchanged and fills the rest with children: two parents
 * each the fittest of `tournament` drawn at random, crossed unit by unit, then
 * each weight and bias nudged at chance `mutationRate` by a normal of standard
 * deviation `mutationSize`. The first `aloneGenerations` play alone on the
 * board, the next `rookieGenerations` 1v1 against the rookie, and the rest, the
 * league, against opponents drawn from the generation's population and the
 * roster, `fourPlayerShare` of the matches with three of them.
 *
 * The next generation is bred after the last one too, so a finished run's
 * checkpoint can carry on to more generations.
 *
 * Defaults (`DEFAULT_SETTINGS`): 50 generations, 10 of them alone; 50 snakes,
 * 2 elites, tournaments of 3, mutation rate 0.1 and size 0.2, initial size 0.5;
 * one hidden layer of 16, tanh; 10 matches a generation at 8 fps; the
 * personality's fitness weights.
 *
 * Nothing reads the clock or `Math.random`: the same settings give the same log
 * and brains, however many workers play them and wherever the run was stopped
 * and carried on from.
 */
export async function train(settings: TrainSettings, options: TrainOptions = {}): Promise<TrainResult> {
  checkSettings(settings);
  const { workers = 1, from, onGeneration, stop } = options;
  if (from && !(Number.isInteger(from.generation) && from.generation >= 0 && from.generation < settings.generations)) {
    throw new Error(`can't carry on from generation ${from.generation} of ${settings.generations}`);
  }
  if (from && from.stage !== stageOf(from.generation, settings)) {
    throw new Error(`the checkpoint says generation ${from.generation} is in the ${from.stage} stage, but the settings put it in the ${stageOf(from.generation, settings)} stage`);
  }
  const pool = workers > 1 ? new Pool(workers) : undefined;
  try {
    const log: GenerationLog[] = [];
    let population = from?.population ?? [];
    let played: { best: Brain; population: Brain[] } | undefined;
    for (let g = from?.generation ?? 0; g < settings.generations; g++) {
      const { rng, initial, stage, fixtures: matches } = drawGeneration(settings, g);
      if (initial) population = initial;
      const fitness = pool ? await pool.fitness(population, matches, settings) : await fitnessHere(population, matches, settings);
      const line = { generation: g, stage, best: Math.max(...fitness), mean: fitness.reduce((a, b) => a + b, 0) / fitness.length };
      played = { best: population[ranked(fitness)[0]], population };
      population = breed(population, fitness, rng, settings);
      log.push(line);
      onGeneration?.(line, played.best, { generation: g + 1, stage: stageOf(g + 1, settings), population });
      if (stop?.()) break;
    }
    return { log, best: played!.best, population: played!.population };
  } finally {
    await pool?.close();
  }
}

/** Every snake's fitness, played on this thread, letting signals in between snakes. */
async function fitnessHere(population: Brain[], matches: Fixture[], settings: TrainSettings) {
  const fitness: number[] = [];
  for (const brain of population) {
    fitness.push(evaluate(brain, matches, population, settings));
    await new Promise((resolve) => setImmediate(resolve));
  }
  return fitness;
}
