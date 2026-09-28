import { GameMode, RULES_VERSION, Rng, mulberry32 } from "snake-colyseus/engine";
import {
  Activation,
  BRAIN_FORMAT,
  BRAIN_FORMAT_VERSION,
  Brain,
  ENCODER_SIZE,
  ENCODER_VERSION,
  Personality,
  brainDecider,
  pickBot
} from "snake-colyseus/bots";
import { MatchOptions, MatchResult, playMatch } from "./match.ts";
import { DELAYS, MODES } from "./gauntlet.ts";

/** The personalities the trainer has a fitness for. */
export const TRAINABLE: readonly Personality[] = ["glutton"];

/** Alone on the board, then 1v1 against the rookie. */
export type Stage = "alone" | "rookie";

/** Everything that decides a run: the same settings always train the same brains. */
export interface TrainSettings {
  personality: Personality;
  /** Every random draw in the run derives from this and the generation: a whole number in [0, 2^32). */
  seed: number;
  /** How many generations in all. */
  generations: number;
  /** How many of them, from the first, are played alone; the rest are against the rookie. */
  aloneGenerations: number;
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
}

export const DEFAULT_SETTINGS: Omit<TrainSettings, "personality" | "seed"> = {
  generations: 50,
  aloneGenerations: 10,
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
  tickBonus: 0.1
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

/** One match every snake in a generation plays: the snake takes seat `seat`, and the rookie the other if there is one. */
export interface Fixture {
  seed: number;
  mode: GameMode;
  delay: number;
  seat: number;
}

/** Throws, saying why, if `settings` can't be trained. */
export function checkSettings(settings: TrainSettings) {
  const { personality, seed, generations, aloneGenerations, population, elites, tournament } = settings;
  if (!TRAINABLE.includes(personality)) {
    throw new Error(`only ${TRAINABLE.join(", ")} can be trained for now, not ${personality}`);
  }
  const whole = (name: string, value: number, min: number) => {
    if (!Number.isInteger(value) || value < min) throw new Error(`${name} must be a whole number of at least ${min}, not ${value}`);
  };
  if (!Number.isInteger(seed) || seed < 0 || seed >= 2 ** 32) throw new Error(`seed must be a whole number in [0, 2^32), not ${seed}`);
  whole("generations", generations, 1);
  whole("aloneGenerations", aloneGenerations, 0);
  if (aloneGenerations > generations) throw new Error(`aloneGenerations (${aloneGenerations}) can't be more than generations (${generations})`);
  whole("population", population, 2);
  whole("elites", elites, 0);
  if (elites >= population) throw new Error(`elites (${elites}) must be fewer than the population (${population})`);
  whole("tournament", tournament, 1);
  whole("matches", settings.matches, 1);
  settings.hidden.forEach((size) => whole("each hidden size", size, 1));
  if (!["tanh", "relu", "sigmoid"].includes(settings.activation)) throw new Error(`activation must be tanh, relu or sigmoid, not ${settings.activation}`);
  if (!(settings.mutationRate >= 0 && settings.mutationRate <= 1)) throw new Error(`mutationRate must be in [0, 1], not ${settings.mutationRate}`);
  for (const name of ["mutationSize", "initialSize", "fps", "foodWeight", "tickBonus"] as const) {
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

/** The generation's matches: seeds, modes, delays (0 to 4) and, against the rookie, seats, all drawn from `rng`. */
export const fixtures = (rng: Rng, stage: Stage, count: number): Fixture[] =>
  Array.from({ length: count }, () => ({
    seed: Math.floor(rng() * 2 ** 32),
    mode: pick(rng, MODES),
    delay: pick(rng, DELAYS),
    seat: stage === "rookie" ? Math.floor(rng() * 2) : 0
  }));

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

/** The stage generation `g` is played in. */
export const stageOf = (g: number, settings: Pick<TrainSettings, "aloneGenerations">): Stage =>
  g < settings.aloneGenerations ? "alone" : "rookie";

/** A snake's fitness over the generation's fixtures. */
export function evaluate(brain: Brain, fixtures: Fixture[], stage: Stage, settings: TrainSettings) {
  const snake = { name: "Candidate", decider: brainDecider(brain) };
  const rookie = pickBot("rookie");
  return gluttonFitness(fixtures.map(({ seed, mode, delay, seat }) => {
    const players = stage === "alone" ? [snake] : seat === 0 ? [snake, rookie] : [rookie, snake];
    const options: MatchOptions = { seed, mode, fps: settings.fps, seats: players.map((player) => ({ player, delay })) };
    return { match: playMatch(options), seat };
  }), settings);
}

/**
 * Neuroevolution: trains a population of encoder-v1 brains for a personality.
 *
 * Each generation `g` draws everything random from `generationRng(seed, g)`:
 * the first generation's brains (`initialSize`), then the generation's
 * `matches` fixtures (seed, mode, delay 0 to 4, and seat), which every snake
 * plays so their fitness can be compared, then the next generation. That keeps
 * the `elites` fittest unchanged and fills the rest with children: two parents
 * each the fittest of `tournament` drawn at random, crossed unit by unit, then
 * each weight and bias nudged at chance `mutationRate` by a normal of standard
 * deviation `mutationSize`. The first `aloneGenerations` play alone on the
 * board; the rest play 1v1 against the rookie at the same delay.
 *
 * Defaults (`DEFAULT_SETTINGS`): 50 generations, 10 of them alone; 50 snakes,
 * 2 elites, tournaments of 3, mutation rate 0.1 and size 0.2, initial size 0.5;
 * one hidden layer of 16, tanh; 10 matches a generation at 8 fps; fitness
 * weights 1 for food and 0.1 a tick survived.
 *
 * Nothing reads the clock or `Math.random`: the same settings give the same log and brains.
 */
export function train(settings: TrainSettings, onGeneration?: (line: GenerationLog) => void): TrainResult {
  checkSettings(settings);
  const log: GenerationLog[] = [];
  let population: Brain[] = [];
  for (let g = 0; ; g++) {
    const rng = generationRng(settings.seed, g);
    if (g === 0) {
      population = Array.from({ length: settings.population }, () =>
        randomBrain(rng, settings.hidden, settings.activation, settings.initialSize));
    }
    const stage = stageOf(g, settings);
    const matches = fixtures(rng, stage, settings.matches);
    const fitness = population.map((brain) => evaluate(brain, matches, stage, settings));
    const line = { generation: g, stage, best: Math.max(...fitness), mean: fitness.reduce((a, b) => a + b, 0) / fitness.length };
    log.push(line);
    onGeneration?.(line);
    if (g === settings.generations - 1) return { log, best: population[ranked(fitness)[0]], population };
    population = breed(population, fitness, rng, settings);
  }
}
