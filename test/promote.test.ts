import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, relative } from "node:path";
import { describe, it } from "node:test";
import { loadRoster } from "snake-colyseus/bots";
import { mulberry32 } from "snake-colyseus/engine";
import { GauntletOptions, GauntletReport } from "../src/gauntlet.ts";
import { bumpMinor, promote, slug } from "../src/promote.ts";
import { createRun, runFiles } from "../src/run.ts";
import { DEFAULT_SETTINGS, TrainSettings, randomBrain } from "../src/train.ts";

const settings: TrainSettings = { ...DEFAULT_SETTINGS, personality: "glutton", seed: 3, generations: 4, aloneGenerations: 1, population: 6, hidden: [4] };
const dummyEntry = { id: "dummy", name: "Dummy", generation: 0, method: "hand-made", brain: "dummy.json" };

const scratch = () => mkdtempSync(join(tmpdir(), "promote-"));
const json = (value: unknown) => `${JSON.stringify(value, null, 2)}\n`;

/** A run folder with generation 3's brain saved, as the trainer would. */
function newRun(): string {
  const run = join(scratch(), "run");
  createRun(run, settings);
  writeFileSync(runFiles(run).generation(3), json(randomBrain(mulberry32(1), [4], "tanh", 0.5)));
  return run;
}

/** The parts of a snake-colyseus checkout `promote` touches, with the dummy in its roster. */
function newCheckout(overrides: Record<string, unknown> = {}): string {
  const dir = scratch();
  writeFileSync(join(dir, "package.json"), json({ private: true, name: "snake-colyseus", version: "1.0.0", engineVersion: "4.0.0", ...overrides }));
  mkdirSync(join(dir, "src/bots/entries"), { recursive: true });
  mkdirSync(join(dir, "src/bots/brains"));
  writeFileSync(join(dir, "src/bots/entries/dummy.json"), json(dummyEntry));
  writeFileSync(join(dir, "src/bots/brains/dummy.json"), json(randomBrain(mulberry32(2), [4], "tanh", 0.5)));
  return dir;
}

/** Every file under the folder, by path and contents. */
const snapshot = (dir: string) => Object.fromEntries(readdirSync(dir, { recursive: true, withFileTypes: true })
  .filter((e) => e.isFile())
  .map((e) => [relative(dir, join(e.parentPath, e.name)), readFileSync(join(e.parentPath, e.name), "utf8")]));

/** A gauntlet that gives the candidate `wins` of its `matches` against the rookie at delay 2. */
const standIn = (wins: number, matches = 10) => (o: GauntletOptions): GauntletReport => ({
  candidate: o.candidate.name,
  baseSeed: o.baseSeed,
  seeds: o.seeds,
  fps: o.fps,
  cells: [{ opponent: "rookie", delay: 2, mode: "timed", matches, wins, losses: matches - wins, draws: 0 }]
});

const base = (run: string, target: string) => ({ run, generation: 3, name: "Nimble Pete", target, seeds: 5, baseSeed: 1, fps: 8 });

describe("slug and bumpMinor", () => {
  it("makes an id from a name", () => {
    assert.equal(slug("Nimble Pete"), "nimble-pete");
    assert.equal(slug("  Zoë_2! "), "zoe-2");
    assert.equal(slug("!!!"), "");
  });

  it("bumps the minor version and resets the patch", () => {
    assert.equal(bumpMinor("4.0.3"), "4.1.0");
    assert.throws(() => bumpMinor("4.0"), /isn't x\.y\.z/);
  });
});

describe("promote", () => {
  it("writes a brain and an entry the package's roster loader reads, and bumps engineVersion by a minor", () => {
    const run = newRun();
    const target = newCheckout();
    const result = promote({ ...base(run, target), gauntlet: standIn(8) });
    assert.equal(result.id, "nimble-pete");
    assert.ok(result.promotion);

    const errors: string[] = [];
    const loaded = loadRoster(join(target, "src/bots"), { error: (m: string) => errors.push(m), warn: (m: string) => errors.push(m) } as never);
    assert.deepEqual(errors, []);
    const entry = loaded.find((e) => e.id === "nimble-pete");
    assert.ok(entry && entry.kind === "brain");
    assert.deepEqual(
      { name: entry.name, personality: entry.personality, generation: entry.generation, method: entry.method },
      { name: "Nimble Pete", personality: "glutton", generation: 3, method: "neuroevolution" }
    );
    assert.deepEqual(entry.brain, JSON.parse(readFileSync(runFiles(run).generation(3), "utf8")));
    assert.equal(JSON.parse(readFileSync(join(target, "package.json"), "utf8")).engineVersion, "4.1.0");
    assert.ok(existsSync(join(run, "promote-nimble-pete.md")));
  });

  it("reads the method from a PPO run: the entry says ppo, with the checkpoint as its generation and the run's personality", () => {
    const run = newRun();
    const ppoSettings = { method: "ppo", personality: "hunter", seed: 3, generations: 4, aloneGenerations: 1, rookieGenerations: 2, leagueGenerations: 1, fps: 8 };
    writeFileSync(runFiles(run).settings, json(ppoSettings));
    const target = newCheckout();
    const { promotion } = promote({ ...base(run, target), gauntlet: standIn(8) });
    assert.ok(promotion);

    const errors: string[] = [];
    const loaded = loadRoster(join(target, "src/bots"), { error: (m: string) => errors.push(m), warn: (m: string) => errors.push(m) } as never);
    assert.deepEqual(errors, []);
    const entry = loaded.find((e) => e.id === "nimble-pete");
    assert.ok(entry && entry.kind === "brain");
    assert.deepEqual({ personality: entry.personality, generation: entry.generation, method: entry.method }, { personality: "hunter", generation: 3, method: "ppo" });
    assert.equal(JSON.parse(readFileSync(join(target, "package.json"), "utf8")).engineVersion, "4.1.0");
    assert.match(promotion.description, /\(ppo, seed 3\)/);
  });

  it("holds the gauntlet's table and verdict in the PR description", () => {
    const run = newRun();
    const { promotion, table } = promote({ ...base(run, newCheckout()), gauntlet: standIn(8) });
    const { description } = promotion!;
    assert.ok(description.includes(table));
    assert.match(description, /won 80\.0%: PASS/);
    assert.match(description, /Nimble Pete/);
    assert.match(description, /"personality": "glutton"/);
    assert.match(description, /run `.*run`/);
    assert.match(description, /npm run promote -- .* 3 --name "Nimble Pete" --seeds 5 --base-seed 1 --fps 8/);
    assert.match(description, /`engine-v\d+\.\d+\.\d+`/);
    assert.equal(readFileSync(join(run, "promote-nimble-pete.md"), "utf8"), description);
  });

  it("writes nothing when the gauntlet fails", () => {
    const run = newRun();
    const target = newCheckout();
    const before = [snapshot(run), snapshot(target)];
    const result = promote({ ...base(run, target), gauntlet: standIn(5) });
    assert.equal(result.verdict.pass, false);
    assert.equal(result.promotion, undefined);
    assert.match(result.table, /won 50\.0%: FAIL/);
    assert.deepEqual([snapshot(run), snapshot(target)], before);
  });

  const refused = (why: RegExp, run: string, target: string, extra: Partial<Parameters<typeof promote>[0]> = {}) => {
    const before = [snapshot(run), snapshot(target)];
    assert.throws(() => promote({ ...base(run, target), gauntlet: standIn(10), ...extra }), why);
    assert.deepEqual([snapshot(run), snapshot(target)], before);
  };

  it("refuses a taken id or name", () => {
    refused(/id "dummy" is taken/, newRun(), newCheckout(), { name: "Dummy" });
    refused(/id "dummy" is taken/, newRun(), newCheckout(), { name: "dummy!" });
    const target = newCheckout();
    writeFileSync(join(target, "src/bots/entries/other.json"), json({ ...dummyEntry, id: "other", name: "Nimble Pete" }));
    refused(/name "Nimble Pete" is taken/, newRun(), target);
    refused(/name "NIMBLE pete" is taken/, newRun(), target, { name: "NIMBLE pete" });
  });

  it("refuses a missing or invalid checkpoint", () => {
    const target = newCheckout();
    const before = snapshot(target);
    refused(/no checkpoint at/, newRun(), target, { generation: 2 });
    const run = newRun();
    writeFileSync(runFiles(run).generation(3), "{ not json");
    refused(/isn't JSON/, run, target);
    writeFileSync(runFiles(run).generation(3), json({ format: "nope" }));
    refused(/isn't a valid brain/, run, target);
    assert.throws(() => promote({ ...base(join(scratch(), "nowhere"), target), gauntlet: standIn(10) }), /isn't a run folder/);
    assert.deepEqual(snapshot(target), before);
  });

  it("refuses a target that isn't a checkout", () => {
    refused(/isn't a snake-colyseus checkout/, newRun(), scratch());
    refused(/isn't a snake-colyseus checkout/, newRun(), newCheckout({ name: "something-else" }));
    const noBots = scratch();
    writeFileSync(join(noBots, "package.json"), json({ name: "snake-colyseus", engineVersion: "4.0.0" }));
    refused(/isn't a snake-colyseus checkout/, newRun(), noBots);
  });

  it("plays the real gauntlet with a tiny seed count", () => {
    const run = newRun();
    const target = newCheckout();
    const result = promote({ ...base(run, target), seeds: 1 });
    assert.ok(result.report.cells.length > 0);
    assert.equal(result.verdict.matches, 4);
    assert.equal(existsSync(join(target, "src/bots/brains/nimble-pete.json")), result.verdict.pass);
  });
});

describe("the promote command", () => {
  it("refuses with exit 1 and writes nothing when the target isn't a checkout", () => {
    const run = newRun();
    const target = scratch();
    const { status, stderr } = spawnSync("npx", ["tsx", "src/cli/promote.ts", run, "3", "--name", "Nimble Pete", "--target", target], { encoding: "utf8" });
    assert.equal(status, 1);
    assert.match(stderr, /Refused, nothing written: .* isn't a snake-colyseus checkout/);
    assert.deepEqual(readdirSync(target), []);
  });
});
