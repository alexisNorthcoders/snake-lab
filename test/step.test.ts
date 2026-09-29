import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { Decider, ENCODER_SIZE, ENCODER_V2_SIZE, deciderFor, encode, encodeV2, pickBot } from "snake-colyseus/bots";
import { MatchOptions, STRAIGHT, StepFacts, SteppedMatch, actionFor, directionFor, playMatch } from "../src/match.ts";

const rookie = pickBot("rookie");
const dummy = pickBot("dummy");
const rookieDecider = deciderFor(rookie);
const dummyDecider = deciderFor(dummy);

const options = (overrides: Partial<MatchOptions> = {}): MatchOptions => ({
  seed: 1,
  seats: [{ player: rookie, delay: 2 }, { player: rookie, delay: 2 }],
  mode: "endless",
  fps: 8,
  ...overrides
});

/** Steps a match with `learner` played by `decider`, and returns it with every step's facts. */
const drive = (opts: MatchOptions, learner: number, decider: Decider, encoder: 1 | 2 = 1) => {
  const match = new SteppedMatch({ ...opts, learner, encoder });
  const facts: StepFacts[] = [];
  while (!match.over) {
    const view = match.view();
    facts.push(match.step(actionFor(view.self.movedDirection, decider(view))));
  }
  return { match, facts };
};

describe("SteppedMatch", () => {
  it("driven by a seat's own decider, gives playMatch's result and events", () => {
    for (const seed of [1, 6]) {
      for (const mode of ["timed", "endless"] as const) {
        for (const learner of [0, 1]) {
          const opts = options({ seed, mode });
          const { match } = drive(opts, learner, rookieDecider);
          assert.deepEqual(match.result, playMatch(opts), `seed ${seed}, ${mode}, learner ${learner}`);
        }
      }
    }
  });

  it("gives the encoder's output for the learner's delayed view, in v1 and v2", () => {
    for (const [encoder, size, encoded] of [[1, ENCODER_SIZE, encode], [2, ENCODER_V2_SIZE, encodeV2]] as const) {
      const match = new SteppedMatch({ ...options({ seed: 6 }), learner: 1, encoder });
      let checked = 0;
      while (!match.over) {
        const observation = match.observe();
        assert.equal(observation.length, size);
        assert.deepEqual(observation, encoded(match.view()));
        match.step(STRAIGHT);
        checked++;
      }
      assert.ok(checked > 1);
    }
  });

  it("sees the other snakes as they were `delay` ticks ago", () => {
    const seats = (delay: number) => [{ player: rookie, delay }, { player: rookie, delay: 0 }];
    const now = new SteppedMatch({ ...options({ seats: seats(0) }), learner: 0 });
    const late = new SteppedMatch({ ...options({ seats: seats(3) }), learner: 0 });
    for (let i = 0; i < 4; i++) {
      now.step(STRAIGHT);
      late.step(STRAIGHT);
    }
    assert.deepEqual(late.view().self, now.view().self);
    assert.notDeepEqual(late.view().others, now.view().others);
  });

  it("turns left, straight and right of the way the snake moved", () => {
    assert.deepEqual(([0, 1, 2] as const).map((a) => directionFor({ x: 1, y: 0 }, a)), ["u", "r", "d"]);
    assert.deepEqual(([0, 1, 2] as const).map((a) => directionFor({ x: 0, y: -1 }, a)), ["l", "u", "r"]);
    assert.deepEqual(([0, 1, 2] as const).map((a) => directionFor({ x: -1, y: 0 }, a)), ["d", "l", "u"]);
    assert.deepEqual(([0, 1, 2] as const).map((a) => directionFor({ x: 0, y: 1 }, a)), ["r", "d", "l"]);
    assert.deepEqual(([0, 1, 2] as const).map((a) => directionFor({ x: 0, y: 0 }, a)), ["u", "r", "d"]);
    assert.equal(actionFor({ x: 1, y: 0 }, "l"), STRAIGHT);
  });

  it("refuses a bad action, a step after the end and an observation with no learner", () => {
    const match = new SteppedMatch({ ...options(), learner: 0 });
    assert.throws(() => match.step(3 as any), /action/);
    assert.throws(() => match.step(undefined), /action/);
    const plain = new SteppedMatch(options());
    assert.throws(() => plain.observe(), /no learner/);
    while (!plain.over) plain.step();
    assert.throws(() => plain.step(), /over/);
    assert.throws(() => new SteppedMatch({ ...options(), learner: 2 }), /learner/);
  });

  describe("facts", () => {
    const events = (opts: MatchOptions) => playMatch(opts).events!;

    it("report the food eaten and the score gained", () => {
      const { match, facts } = drive(options({ seed: 20, mode: "timed" }), 0, rookieDecider);
      const ate = events(options({ seed: 20, mode: "timed" })).filter((e) => e.kind === "ate" && e.player === "seat0");
      assert.ok(ate.length > 0, "the seed should have seat0 eat");
      assert.equal(facts.reduce((sum, f) => sum + f.ate, 0), ate.length);
      assert.equal(facts.reduce((sum, f) => sum + f.scoreGained, 0), match.result!.players[0].score);
      facts.forEach((f, i) => assert.equal(f.tick, i + 1));
    });

    it("report a death with its cause and killer, and the kill", () => {
      // Seed 6, rookie against rookie: seat 0 runs into seat 1's body.
      const opts = options({ seed: 6 });
      const victim = drive(opts, 0, rookieDecider);
      const dead = victim.facts.find((f) => f.death);
      assert.deepEqual(dead?.death, { cause: "body", by: "seat1" });
      assert.equal(dead?.alive, false);
      assert.equal(dead?.kills, 0);

      const killer = drive(opts, 1, rookieDecider);
      assert.equal(killer.facts.reduce((sum, f) => sum + f.kills, 0), 1);
      assert.equal(killer.facts.find((f) => f.kills)?.alive, true);
    });

    it("report a mutual head-on as a death for both and a kill for neither", () => {
      // Seed 1, endless: the two rookies meet head-on on tick 18.
      for (const learner of [0, 1]) {
        const { facts } = drive(options({ seed: 1 }), learner, rookieDecider);
        const last = facts.at(-1)!;
        assert.equal(last.death?.cause, "head-on");
        assert.equal(last.death?.by, `seat${1 - learner}`);
        assert.equal(facts.reduce((sum, f) => sum + f.kills, 0), 0);
      }
    });

    it("report a death of one's own doing, with no killer", () => {
      const seats = [{ player: dummy, delay: 2 }, { player: rookie, delay: 2 }];
      const { facts } = drive(options({ seed: 4, seats }), 0, dummyDecider);
      const dead = facts.find((f) => f.death);
      assert.deepEqual(dead?.death, { cause: "self" });
    });

    it("report the result and why the round ended, on the last step only", () => {
      const { match, facts } = drive(options({ seed: 6 }), 1, rookieDecider);
      const last = facts.at(-1)!;
      assert.equal(last.over, true);
      assert.equal(last.outcome, "won");
      assert.equal(last.reason, match.result!.result.reason);
      assert.ok(facts.slice(0, -1).every((f) => !f.over && f.outcome === undefined && f.reason === undefined));

      assert.equal(drive(options({ seed: 6 }), 0, rookieDecider).facts.at(-1)!.outcome, "lost");
      assert.equal(drive(options({ seed: 1 }), 0, rookieDecider).facts.at(-1)!.outcome, "drawn");
      const timed = drive(options({ seed: 3, mode: "timed", fps: 0.05 }), 0, rookieDecider);
      assert.equal(timed.facts.at(-1)!.reason, "time-up");
    });
  });
});
