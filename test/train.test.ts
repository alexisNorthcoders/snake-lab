import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";
import { mulberry32 } from "snake-colyseus/engine";
import { brainProblems } from "snake-colyseus/bots";
import { loadCandidate } from "../src/gauntlet.ts";
import { MatchResult } from "../src/match.ts";
import {
  DEFAULT_SETTINGS,
  TrainSettings,
  breed,
  checkSettings,
  crossover,
  fixtures,
  gluttonFitness,
  mutate,
  randomBrain,
  ranked,
  train
} from "../src/train.ts";

const settings = (overrides: Partial<TrainSettings> = {}): TrainSettings =>
  ({ ...DEFAULT_SETTINGS, personality: "glutton", seed: 1, ...overrides });

const short = settings({ generations: 3, aloneGenerations: 1, population: 6, matches: 2, hidden: [4] });

const brains = (count: number, seed = 1) => {
  const rng = mulberry32(seed);
  return Array.from({ length: count }, () => randomBrain(rng, [16], "tanh", 0.5));
};

const player = (id: string, score: number) => ({ id, name: id, score, length: 1, decisionErrors: 0 });

describe("breeding", () => {
  it("starts from small random brains that are valid", () => {
    brains(5).forEach((brain) => assert.deepEqual(brainProblems(brain), []));
    assert.deepEqual(randomBrain(mulberry32(1), [16], "tanh", 0.5).sizes, [23, 16, 3]);
  });

  it("crosses whole units: each unit's weights and bias come from one parent", () => {
    const [a, b] = brains(2);
    const child = crossover(a, b, mulberry32(3));
    assert.deepEqual(brainProblems(child), []);
    child.layers.forEach((layer, l) => layer.biases.forEach((bias, j) => {
      const from = [a, b].find((p) => p.layers[l].biases[j] === bias)!;
      assert.ok(from, `layer ${l} unit ${j}'s bias is from neither parent`);
      assert.deepEqual(layer.weights[j], from.layers[l].weights[j]);
    }));
    const fromA = child.layers[0].biases.filter((bias, j) => bias === a.layers[0].biases[j]).length;
    assert.ok(fromA > 0 && fromA < 16, "the hidden units come from both parents");
  });

  it("mutates some weights by small nudges, leaves the rest, and keeps the brain valid", () => {
    const [brain] = brains(1);
    const mutant = mutate(brain, mulberry32(4), 0.1, 0.2);
    assert.deepEqual(brainProblems(mutant), []);
    const before = brain.layers.flatMap((l) => [...l.weights.flat(), ...l.biases]);
    const after = mutant.layers.flatMap((l) => [...l.weights.flat(), ...l.biases]);
    const changed = after.filter((w, i) => w !== before[i]).length;
    assert.ok(changed > 0 && changed < before.length / 4, `${changed} of ${before.length} changed`);
    assert.ok(after.every((w, i) => Math.abs(w - before[i]) < 2));
    assert.notDeepEqual(brain, mutant);
  });

  it("keeps the population size, copies the elites unchanged, and breeds valid brains", () => {
    const population = brains(10);
    const fitness = [5, 90, 30, 70, 10, 0, 20, 60, 40, 80];
    const next = breed(population, fitness, mulberry32(5), { elites: 3, tournament: 3, mutationRate: 0.1, mutationSize: 0.2 });
    assert.equal(next.length, 10);
    assert.deepEqual(next.slice(0, 3), [population[1], population[9], population[3]]);
    next.forEach((brain) => assert.deepEqual(brainProblems(brain), []));
    assert.deepEqual(ranked(fitness).slice(0, 3), [1, 9, 3]);
  });
});

describe("gluttonFitness", () => {
  const match = (overrides: Partial<MatchResult>): MatchResult => ({
    result: { reason: "last-standing" },
    ticks: 300,
    players: [player("seat0", 120), player("seat1", 40)],
    events: [],
    ...overrides
  });

  it("is the food score plus a bonus a tick survived, averaged over the snake's matches", () => {
    const died = match({ events: [{ kind: "died", player: "seat1", cause: "wall", tick: 100 } as never] });
    const lived = match({ ticks: 200, players: [player("seat0", 0), player("seat1", 70)] });
    const weights = { foodWeight: 2, tickBonus: 0.5 };
    // seat 1: died on tick 100 with 40, then lived all 200 ticks with 70.
    assert.equal(gluttonFitness([{ match: died, seat: 1 }, { match: lived, seat: 1 }], weights),
      ((2 * 40 + 0.5 * 100) + (2 * 70 + 0.5 * 200)) / 2);
    // seat 0 lived the whole 300 ticks.
    assert.equal(gluttonFitness([{ match: died, seat: 0 }], weights), 2 * 120 + 0.5 * 300);
  });
});

describe("fixtures", () => {
  it("draws both modes and delays 0 to 4 from the seed, and seats only against the rookie", () => {
    const drawn = fixtures(mulberry32(6), "rookie", 200);
    assert.deepEqual(new Set(drawn.map((f) => f.mode)), new Set(["timed", "endless"]));
    assert.deepEqual([...new Set(drawn.map((f) => f.delay))].sort(), [0, 1, 2, 3, 4]);
    assert.deepEqual(new Set(drawn.map((f) => f.seat)), new Set([0, 1]));
    assert.ok(fixtures(mulberry32(6), "alone", 50).every((f) => f.seat === 0));
    assert.deepEqual(fixtures(mulberry32(6), "rookie", 200), drawn);
  });
});

describe("checkSettings", () => {
  it("refuses personalities it has no fitness for, and settings that can't be trained", () => {
    assert.throws(() => checkSettings(settings({ personality: "hunter" })), /only glutton can be trained/);
    assert.throws(() => checkSettings(settings({ elites: 50 })), /elites/);
    assert.throws(() => checkSettings(settings({ aloneGenerations: 60 })), /aloneGenerations/);
    assert.doesNotThrow(() => checkSettings(settings()));
  });
});

describe("train", () => {
  it("gives identical logs and brains from the same seed, and different ones from another", () => {
    const first = train(short);
    const again = train(short);
    const other = train({ ...short, seed: 2 });
    assert.deepEqual(again, first);
    assert.notDeepEqual(other.log, first.log);
    assert.notDeepEqual(other.best, first.best);
  });

  it("logs each generation's best and mean fitness and stage, and ends with a valid brain", () => {
    const lines: unknown[] = [];
    const { log, best, population } = train(short, (line) => lines.push(line));
    assert.deepEqual(lines, log);
    assert.deepEqual(log.map((l) => [l.generation, l.stage]), [[0, "alone"], [1, "rookie"], [2, "rookie"]]);
    log.forEach((l) => assert.ok(l.best >= l.mean));
    assert.equal(population.length, short.population);
    assert.deepEqual(brainProblems(best), []);
  });

  it("learns against the rookie: the best fitness goes up from the first generation to the last", () => {
    const { log } = train(settings({ seed: 5, generations: 5, aloneGenerations: 0, population: 10, matches: 3 }));
    assert.ok(log.at(-1)!.best > log[0].best, JSON.stringify(log));
  });
});

describe("the train command", () => {
  const cli = (script: string, ...args: string[]) =>
    spawnSync(process.execPath, ["--import", "tsx", `src/cli/${script}.ts`, ...args], { encoding: "utf8" });

  it("writes the settings, a log line a generation as it prints them, and a best brain that plays", () => {
    const run = join(mkdtempSync(join(tmpdir(), "train-")), "run");
    const trained = cli("train", "--personality", "glutton", "--run", run, "--seed", "5",
      "--generations", "2", "--alone", "1", "--population", "4", "--matches", "1", "--hidden", "4");
    assert.equal(trained.status, 0, trained.stderr);
    const log = readFileSync(join(run, "log.jsonl"), "utf8").trim().split("\n");
    assert.deepEqual(log.map((line) => JSON.parse(line).stage), ["alone", "rookie"]);
    assert.ok(log.every((line) => trained.stdout.includes(line)));
    assert.equal(JSON.parse(readFileSync(join(run, "settings.json"), "utf8")).seed, 5);

    const best = join(run, "best.json");
    assert.doesNotThrow(() => loadCandidate(best));
    const played = cli("match", "--players", `${best},rookie`, "--seed", "3");
    assert.equal(played.status, 0, played.stderr);
    assert.match(played.stdout, /best\.json/);
  });

  it("refuses a personality it can't train yet, writing nothing", () => {
    const run = join(mkdtempSync(join(tmpdir(), "train-")), "run");
    const refused = cli("train", "--personality", "hunter", "--run", run);
    assert.equal(refused.status, 1);
    assert.match(refused.stderr, /only glutton can be trained for now, not hunter/);
    assert.equal(existsSync(run), false);
  });
});
