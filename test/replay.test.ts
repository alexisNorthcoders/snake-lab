import assert from "node:assert/strict";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";
import { type Decider, brainDecider, encode, layerValues } from "snake-colyseus/bots";
import { gameStream, readGenerationBest, replayable, sampleFixtures } from "../src/dashboard/replay.ts";
import { networkLayout } from "../src/dashboard/network.ts";
import { playMatch } from "../src/match.ts";
import { createRun } from "../src/run.ts";
import { trainRun } from "../src/run.ts";
import { DEFAULT_SETTINGS, type Fixture, type TrainSettings, drawGeneration, evaluate, fixtureOptions, fixtures, generationRng, randomBrain, stageOf } from "../src/train.ts";

const settings: TrainSettings = { ...DEFAULT_SETTINGS, personality: "glutton", seed: 5, generations: 6, aloneGenerations: 2, rookieGenerations: 2, population: 4, matches: 6, hidden: [4] };

describe("drawGeneration", () => {
  it("draws what the trainer's own draws give, in every stage and for generation 0", () => {
    for (const g of [0, 1, 2, 3, 4, 5]) {
      const rng = generationRng(settings.seed, g);
      const initial = g === 0 ? Array.from({ length: settings.population }, () => randomBrain(rng, settings.hidden, settings.activation, settings.initialSize)) : undefined;
      const stage = stageOf(g, settings);
      const drawn = drawGeneration(settings, g);
      assert.equal(drawn.stage, stage);
      assert.deepEqual(drawn.initial, initial);
      assert.deepEqual(drawn.fixtures, fixtures(rng, stage, settings));
      assert.equal(drawn.rng(), rng(), "the generator is left in the same place for breeding");
    }
  });

  it("draws different fixtures for different generations", () => {
    assert.notDeepEqual(drawGeneration(settings, 2).fixtures, drawGeneration(settings, 3).fixtures);
  });
});

describe("sampleFixtures", () => {
  it("keeps every alone and rookie fixture", () => {
    for (const g of [0, 1, 2, 3]) {
      assert.deepEqual(sampleFixtures(settings, g).map(({ fixture }) => fixture), drawGeneration(settings, g).fixtures);
    }
  });

  it("leaves out league fixtures against a population opponent and keeps the rest, with their places", () => {
    const league = { ...settings, matches: 40, fourPlayerShare: 0.5 };
    const all = drawGeneration(league, 4).fixtures;
    const sample = sampleFixtures(league, 4);
    assert.ok(all.some((f) => f.opponents.some((o) => "population" in o)), "some fixtures have a population opponent");
    assert.ok(sample.length > 0 && sample.length < all.length);
    assert.deepEqual(sample.map(({ fixture }) => fixture), all.filter((f) => f.opponents.every((o) => "roster" in o)));
    sample.forEach(({ fixture, index }) => assert.deepEqual(all[index], fixture));
  });

  it("stops at the limit", () => {
    assert.equal(sampleFixtures(settings, 0, 2).length, 2);
  });

  it("replayable drops only fixtures with a population opponent", () => {
    const base = { seed: 1, mode: "endless" as const, seat: 0, delays: [0, 0] };
    const a: Fixture = { ...base, opponents: [{ roster: "rookie" }] };
    const b: Fixture = { ...base, opponents: [{ population: 1 }] };
    assert.deepEqual(replayable([a, b]), [a]);
  });
});

describe("replaying a generation's best", () => {
  it("gives the fitness the trainer logged when fed through the same fitness (alone and rookie stages)", async () => {
    const run = join(mkdtempSync(join(tmpdir(), "replay-")), "run");
    const small = { ...settings, generations: 4 };
    createRun(run, small);
    const result = await trainRun(run);
    assert.ok(result);
    for (const line of result.log) {
      const brain = readGenerationBest(run, line.generation)!;
      const { fixtures: played } = drawGeneration(small, line.generation);
      const replayed = evaluate(brain, sampleFixtures(small, line.generation).map(({ fixture }) => fixture), [], small);
      assert.equal(replayed, line.best, `generation ${line.generation}`);
      assert.equal(played.length, small.matches);
    }
  });

  it("replays a league generation's best in its roster-only fixtures, scored as the trainer scores them", async () => {
    const run = join(mkdtempSync(join(tmpdir(), "replay-")), "run");
    const small = { ...settings, generations: 5 };
    createRun(run, small);
    await trainRun(run);
    const g = 4;
    assert.equal(stageOf(g, small), "league");
    const brain = readGenerationBest(run, g)!;
    const { fixtures: played } = drawGeneration(small, g);
    const kept = sampleFixtures(small, g).map(({ fixture }) => fixture);
    assert.deepEqual(kept, replayable(played));
    assert.ok(kept.every(({ opponents }) => opponents.every((o) => "roster" in o)));
    if (kept.length > 0) assert.ok(Number.isFinite(evaluate(brain, kept, [], small)));
  });

  it("has no brain for a generation not saved yet", () => {
    const run = join(mkdtempSync(join(tmpdir(), "replay-")), "run");
    createRun(run, settings);
    assert.equal(readGenerationBest(run, 0), undefined);
  });
});

describe("gameStream", () => {
  const brain = randomBrain(generationRng(1, 0), [4], "tanh", 0.5);
  for (const g of [0, 2]) {
    it(`starts, ends with the match's result and matches playMatch (generation ${g})`, () => {
      const { fixture, index } = sampleFixtures(settings, g)[0];
      const messages = [...gameStream(fixture, index, brain, settings.fps)];
      const match = playMatch(fixtureOptions(fixture, { name: "Candidate", decider: brainDecider(brain) }, [], settings.fps));
      const first = messages[0], last = messages[messages.length - 1];
      assert.equal(first.kind, "start");
      assert.equal(last.kind, "end");
      if (first.kind !== "start" || last.kind !== "end") return;
      assert.equal(first.game.index, index);
      assert.deepEqual(first.game.delays, fixture.delays);
      assert.deepEqual(last.result, match.result);
      assert.equal(last.ticks, match.ticks);
      assert.deepEqual(last.players.map((p) => p.score), match.players.map((p) => p.score));
      const frames = messages.filter((m) => m.kind === "tick").map((m) => (m as { frame: any }).frame);
      assert.equal(frames.length, match.ticks + 1);
      assert.deepEqual(frames.map((f) => f.tick), Array.from({ length: match.ticks + 1 }, (_, i) => i));
      assert.deepEqual(frames.flatMap((f) => f.events), match.events!.map((e) => e));
      assert.deepEqual(frames[frames.length - 1].snakes.map((s: any) => s.score), match.players.map((p) => p.score));
    });
  }

  it("doesn't change the match: the same result with and without watching", () => {
    const { fixture } = sampleFixtures(settings, 2)[0];
    const options = fixtureOptions(fixture, { name: "Candidate", decider: brainDecider(brain) }, [], settings.fps);
    assert.deepEqual(playMatch({ ...options, onTick: () => {} }), playMatch(options));
  });
});

describe("gameStream's activations", () => {
  const brain = randomBrain(generationRng(1, 0), [6, 5], "tanh", 0.8);

  it("sends, for each tick, the package's layer values for the snake's view, and marks the output that won", () => {
    const { fixture, index } = sampleFixtures(settings, 2)[0];
    const views: Parameters<Decider>[0][] = [];
    const decide = brainDecider(brain);
    const spy: Decider = (view) => { views.push(structuredClone(view)); return decide(view); };
    const options = fixtureOptions(fixture, { name: "Candidate", decider: spy }, [], settings.fps);
    // our seat is the only one whose decider is the spy; the others' views aren't recorded
    const ticks = [...gameStream(fixture, index, brain, settings.fps)].filter((m) => m.kind === "tick") as { frame: { tick: number }; activations?: { values: number[][]; chosen: number } }[];
    playMatch(options);
    assert.equal(ticks[0].activations, undefined, "the board as dealt has no decision behind it");
    const withValues = ticks.filter((t) => t.activations);
    assert.equal(withValues.length, views.length, "one set of values for each decision the snake made");
    withValues.forEach((t, i) => {
      const expected = layerValues(brain, encode(views[i]));
      assert.deepEqual(t.activations!.values, expected);
      assert.equal(t.frame.tick, i + 1);
      const out = expected[expected.length - 1];
      assert.equal(t.activations!.chosen, out.indexOf(Math.max(...out)));
    });
  });

  it("has values for every layer of the brain", () => {
    const { fixture, index } = sampleFixtures(settings, 0)[0];
    const tick = [...gameStream(fixture, index, brain, settings.fps)].find((m) => m.kind === "tick" && m.activations);
    assert.ok(tick && tick.kind === "tick");
    assert.deepEqual(tick.activations!.values.map((v) => v.length), brain.sizes);
  });
});

describe("networkLayout", () => {
  it("lays a 23 → 16 → 3 brain out in three columns with an edge for every weight", () => {
    const layout = networkLayout(randomBrain(generationRng(1, 0), [16], "tanh", 0.5));
    assert.deepEqual(layout.columns.map((c) => c.labels.length), [23, 16, 3]);
    assert.equal(layout.columns[0].labels[0], "blocked left 1");
    assert.equal(layout.columns[0].labels[22], "drains left");
    assert.deepEqual(layout.columns[2].labels, ["left", "straight", "right"]);
    assert.equal(layout.edges.length, 23 * 16 + 16 * 3);
    assert.equal(layout.edges.filter((e) => e.layer === 0).length, 23 * 16);
    assert.ok(layout.edges.every((e) => e.from < layout.columns[e.layer].labels.length && e.to < layout.columns[e.layer + 1].labels.length));
  });

  it("lays a brain with two hidden layers out in four columns, edges carrying the weights", () => {
    const brain = randomBrain(generationRng(2, 0), [8, 5], "relu", 0.5);
    const layout = networkLayout(brain);
    assert.deepEqual(layout.columns.map((c) => c.labels.length), [23, 8, 5, 3]);
    assert.equal(layout.edges.length, 23 * 8 + 8 * 5 + 5 * 3);
    for (const { layer, from, to, weight } of layout.edges) assert.equal(weight, brain.layers[layer].weights[to][from]);
    assert.deepEqual(new Set(layout.edges.map((e) => e.layer)), new Set([0, 1, 2]));
  });
});
