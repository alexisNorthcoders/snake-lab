import assert from "node:assert/strict";
import { type ChildProcessByStdio, spawn } from "node:child_process";
import { after, describe, it } from "node:test";
import { Readable, Writable } from "node:stream";
import { pickBot } from "snake-colyseus/bots";
import { FACT_COUNT, FACT_FIELDS, type Frame, FrameReader, encodeFrame } from "../src/env.ts";
import { mulberry32 } from "snake-colyseus/engine";
import { STRAIGHT, SteppedMatch } from "../src/match.ts";

/** An `env` subprocess, spoken to through its protocol. */
class Client {
  private child: ChildProcessByStdio<Writable, Readable, null>;
  private reader = new FrameReader();
  private waiting: ((frame: Frame) => void)[] = [];
  private queue: Frame[] = [];

  constructor() {
    this.child = spawn(process.execPath, ["--import", "tsx", "src/cli/env.ts"], { stdio: ["pipe", "pipe", "inherit"] });
    this.child.stdout.on("data", (chunk: Buffer) => {
      for (const frame of this.reader.push(chunk)) (this.waiting.shift() ?? ((f) => this.queue.push(f)))(frame);
    });
  }

  request(header: Frame["header"], payload = Buffer.alloc(0)): Promise<Frame> {
    const reply = new Promise<Frame>((resolve) => {
      const queued = this.queue.shift();
      queued ? resolve(queued) : this.waiting.push(resolve);
    });
    this.child.stdin.write(encodeFrame({ header, payload }));
    return reply;
  }

  async close() {
    const exited = new Promise((resolve) => this.child.once("exit", resolve));
    await this.request({ type: "close" });
    await exited;
  }
}

/** A batch's payload split into its facts (`facts[match][field]`) and observations. */
const split = ({ header, payload }: Frame) => {
  const n = header.matches as number;
  const bytes = payload.buffer.slice(payload.byteOffset, payload.byteOffset + payload.byteLength);
  const facts = new Int32Array(bytes, 0, n * FACT_COUNT);
  const observations = new Float32Array(bytes, n * FACT_COUNT * 4, n * header.obsSize);
  const field = (match: number, name: (typeof FACT_FIELDS)[number]) => facts[match * FACT_COUNT + FACT_FIELDS.indexOf(name)];
  return { n, obsSize: header.obsSize as number, facts, observations, field };
};

const config = (overrides = {}) => ({
  type: "reset",
  seed: 5,
  encoder: 2,
  fps: 8,
  matches: [
    { opponents: [] },
    { opponents: ["rookie"], mode: "endless", learnerDelay: [0, 4], opponentDelay: [0, 4], seat: "random" },
    { opponents: ["rookie", "dummy", "rookie"], mode: "mixed", seat: 3 }
  ],
  ...overrides
});

const actions = (n: number, step: number) => Buffer.from(Array.from({ length: n }, (_, i) => (i + step) % 3));

describe("the env subprocess", () => {
  const clients: Client[] = [];
  const client = () => clients[clients.push(new Client()) - 1];
  after(() => Promise.all(clients.map((c) => c.close())));

  it("resets many matches at once and gives each one's first observation", async () => {
    const reply = await client().request(config());
    assert.equal(reply.header.type, "batch");
    assert.equal(reply.header.protocol, 1);
    const { n, obsSize, observations, facts } = split(reply);
    assert.equal(n, 3);
    assert.equal(obsSize, 328);
    assert.equal(observations.length, 3 * 328);
    assert.equal(reply.payload.length, n * FACT_COUNT * 4 + n * 328 * 4);
    assert.ok(observations.some((v) => v !== 0));
    assert.equal(facts[FACT_FIELDS.indexOf("alive")], 1);
  });

  it("gives the observation of a stepped match, as float32", async () => {
    const reply = await client().request(config({ matches: [{ opponents: ["rookie"], mode: "endless", seat: 1 }], seed: 9 }));
    const { observations } = split(reply);
    assert.equal(observations.length, 328);
    // The env draws the match's seed from the batch's: the first draw of mulberry32(9).
    const { mulberry32 } = await import("snake-colyseus/engine");
    const seed = Math.floor(mulberry32(9)() * 2 ** 32);
    const rookie = pickBot("rookie");
    const match = new SteppedMatch({
      seed, mode: "endless", fps: 8, learner: 1, encoder: 2,
      seats: [{ player: rookie, delay: 2 }, { player: rookie, delay: 2 }]
    });
    assert.deepEqual(Array.from(observations), Array.from(Float32Array.from(match.observe())));
  });

  it("steps them together, and resets a match that ended on a new seed", async () => {
    const c = client();
    let batch = split(await c.request(config({ matches: [{ opponents: ["rookie"], mode: "endless" }, { opponents: [], mode: "endless" }] })));
    let ended = 0;
    let previous: Float32Array | undefined;
    for (let step = 0; step < 400; step++) {
      const reply = await c.request({ type: "step" }, actions(2, step));
      assert.equal(reply.header.type, "batch");
      batch = split(reply);
      for (let i = 0; i < 2; i++) {
        if (!batch.field(i, "ended")) continue;
        ended++;
        assert.ok(batch.field(i, "outcome") > 0 && batch.field(i, "reason") > 0, "an ended match says how");
        assert.ok(batch.field(i, "ticks") > 0);
      }
      previous = batch.observations;
    }
    assert.ok(ended >= 2, `matches should have ended and restarted, ${ended} did`);
    assert.ok(previous!.every((v) => Number.isFinite(v)));
  });

  it("pairs an ended match's last facts with the next match's first observation, as a stepped match does", async () => {
    const c = client();
    const rng = mulberry32(9);
    const make = () => new SteppedMatch({
      seed: Math.floor(rng() * 2 ** 32),
      seats: [{ player: { name: "learner", decider: () => "r" }, delay: 2 }],
      mode: "endless", fps: 8, learner: 0, encoder: 2, keepEvents: false
    });
    let ref = make();
    const first = split(await c.request(config({ seed: 9, matches: [{ opponents: [], mode: "endless" }] })));
    assert.deepEqual([...first.observations], ref.observe().map(Math.fround));
    let ended = 0;
    for (let step = 0; step < 300 && ended < 2; step++) {
      const action = (step % 5 === 4 ? 0 : STRAIGHT) as 0 | 1;
      const batch = split(await c.request({ type: "step" }, Buffer.from([action])));
      const f = ref.step(action);
      assert.equal(batch.field(0, "ticks"), f.tick);
      assert.equal(batch.field(0, "ate"), f.ate);
      assert.equal(batch.field(0, "scoreGained"), f.scoreGained);
      assert.equal(batch.field(0, "alive"), f.alive ? 1 : 0);
      const isEnd = !f.alive || f.over;
      assert.equal(batch.field(0, "ended"), isEnd ? 1 : 0);
      if (isEnd) {
        ended++;
        ref = make();
      }
      assert.deepEqual([...batch.observations], ref.observe().map(Math.fround));
    }
    assert.ok(ended >= 1, "the reference match should have ended");
  });

  it("ends a match when the learner dies, unless told not to", async () => {
    const run = async (endOnDeath: boolean) => {
      const c = client();
      await c.request(config({ endOnDeath, matches: [{ opponents: ["rookie"], mode: "endless", seat: 0 }] }));
      for (let step = 0; step < 200; step++) {
        const batch = split(await c.request({ type: "step" }, Buffer.from([STRAIGHT])));
        if (!batch.field(0, "alive")) return { ended: batch.field(0, "ended"), reason: batch.field(0, "reason"), outcome: batch.field(0, "outcome") };
      }
      throw new Error("walking straight should die");
    };
    const early = await run(true);
    assert.equal(early.ended, 1);
    assert.equal(early.outcome, 2);
    const late = await run(false);
    // The rookie may have died with it in the same tick; otherwise the round runs on.
    assert.ok(late.ended === 0 || late.reason !== 3);
  });

  it("gives the same bytes for the same seed and actions", async () => {
    const play = async () => {
      const c = new Client();
      const out: Buffer[] = [(await c.request(config())).payload];
      for (let step = 0; step < 150; step++) out.push((await c.request({ type: "step" }, actions(3, step))).payload);
      await c.close();
      return Buffer.concat(out);
    };
    const first = await play();
    assert.ok(first.equals(await play()));
    const other = new Client();
    const different = (await other.request(config({ seed: 6 }))).payload;
    await other.close();
    assert.ok(!different.equals(first.subarray(0, different.length)));
  });

  it("answers a bad message with an error and carries on", async () => {
    const c = client();
    assert.match((await c.request({ type: "step" })).header.message, /before reset/);
    assert.match((await c.request({ type: "nope" })).header.message, /unknown message/);
    assert.match((await c.request(config({ matches: [{ opponents: ["nobody-here"] }] }))).header.message, /nobody-here/);
    assert.match((await c.request(config({ encoder: 3 }))).header.message, /encoder/);
    await c.request(config());
    assert.match((await c.request({ type: "step" }, Buffer.from([0]))).header.message, /3 actions/);
    assert.equal((await c.request({ type: "step" }, Buffer.from([0, 1, 5]))).header.type, "error");
  });
});
