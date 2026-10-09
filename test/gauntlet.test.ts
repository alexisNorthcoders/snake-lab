import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";
import { dummyBrain, pickBot, roster } from "snake-colyseus/bots";
import {
  BAR,
  type GauntletCell,
  type GauntletReport,
  loadCandidate,
  opponentsFor,
  rates,
  runGauntlet,
  verdict
} from "../src/gauntlet.ts";

const rookie = pickBot("rookie");
const dummy = pickBot("dummy");

const small = { baseSeed: 5, seeds: 2, fps: 8, delays: [0, 2], modes: ["timed", "endless"] as const };

const cell = (overrides: Partial<GauntletCell>): GauntletCell =>
  ({ opponent: "rookie", delay: 2, mode: "timed", matches: 0, wins: 0, losses: 0, draws: 0, ...overrides });

const report = (cells: GauntletCell[]): GauntletReport => ({ candidate: "Test", baseSeed: 1, seeds: 50, fps: 8, cells });

const tempFile = (contents: string) => {
  const path = join(mkdtempSync(join(tmpdir(), "gauntlet-")), "brain.json");
  writeFileSync(path, contents);
  return path;
};

const cli = (...args: string[]) =>
  spawnSync(process.execPath, ["--import", "tsx", "src/cli/gauntlet.ts", ...args], { encoding: "utf8" });

describe("runGauntlet", () => {
  it("gives an identical report for the same candidate, base seed and N", () => {
    const first = runGauntlet({ candidate: dummy, opponents: [rookie], ...small });
    const again = runGauntlet({ candidate: dummy, opponents: [rookie], ...small });
    assert.deepEqual(again, first);
  });

  it("gives a different report from a different base seed", () => {
    const first = runGauntlet({ candidate: dummy, opponents: [rookie], ...small, seeds: 5 });
    const other = runGauntlet({ candidate: dummy, opponents: [rookie], ...small, seeds: 5, baseSeed: 500 });
    assert.notDeepEqual(other.cells, first.cells);
  });

  it("plays 2 × N matches in every cell of opponent, delay and mode, and their rates add up to 1", () => {
    const result = runGauntlet({ candidate: rookie, opponents: [rookie, dummy], ...small });
    assert.equal(result.cells.length, 2 * 2 * 2);
    assert.deepEqual(result.cells.map(({ opponent, delay, mode }) => `${opponent} ${delay} ${mode}`), [
      "rookie 0 timed", "rookie 0 endless", "rookie 2 timed", "rookie 2 endless",
      "dummy 0 timed", "dummy 0 endless", "dummy 2 timed", "dummy 2 endless"
    ]);
    result.cells.forEach((c) => {
      assert.equal(c.matches, 2 * small.seeds);
      assert.equal(c.wins + c.losses + c.draws, c.matches);
      const { win, loss, draw } = rates(c);
      assert.ok(Math.abs(win + loss + draw - 1) < 1e-9);
    });
  });

  it("plays each seed with the candidate in each seat", () => {
    const seated: string[] = [];
    const watcher = { name: "Watcher", decider: () => "r" as const };
    runGauntlet({
      candidate: watcher, opponents: [rookie], baseSeed: 9, seeds: 1, fps: 8, delays: [2], modes: ["timed"],
      onMatch: (options) => seated.push(`${options.seed} ${options.seats.map((s) => `${s.player.name}@${s.delay}`).join(",")}`)
    });
    assert.deepEqual(seated, ["9 Watcher@2,Rookie@2", "9 Rookie@2,Watcher@2"]);
  });

  it("counts the rookie's wins over a snake that runs into the wall as losses, from either seat", () => {
    const wall = { name: "Wall", decider: () => "u" as const };
    const result = runGauntlet({ candidate: wall, opponents: [rookie], ...small, delays: [0] });
    result.cells.forEach((c) => assert.deepEqual([c.wins, c.losses, c.draws], [0, c.matches, 0]));
  });
});

describe("verdict", () => {
  it("passes a candidate just above the bar against the rookie at delay 2, over both modes", () => {
    const v = verdict(report([
      cell({ mode: "timed", matches: 100, wins: 61, losses: 39 }),
      cell({ mode: "endless", matches: 100, wins: 60, losses: 40 })
    ]));
    assert.equal(v.bar, BAR);
    assert.equal(v.winRate, 0.605);
    assert.equal(v.pass, true);
  });

  it("passes a candidate exactly at the bar", () => {
    assert.equal(verdict(report([cell({ matches: 100, wins: 60, losses: 40 })])).pass, true);
  });

  it("fails a candidate just below the bar", () => {
    const v = verdict(report([
      cell({ mode: "timed", matches: 100, wins: 60, losses: 40 }),
      cell({ mode: "endless", matches: 100, wins: 59, losses: 41 })
    ]));
    assert.equal(v.winRate, 0.595);
    assert.equal(v.pass, false);
  });

  it("counts draws as not winning", () => {
    const v = verdict(report([cell({ matches: 100, wins: 59, draws: 41 })]));
    assert.equal(v.winRate, 0.59);
    assert.equal(v.pass, false);
  });

  it("only counts the rookie at delay 2", () => {
    const v = verdict(report([
      cell({ matches: 100, wins: 10, losses: 90 }),
      cell({ delay: 1, matches: 100, wins: 100 }),
      cell({ opponent: "dummy", matches: 100, wins: 100 })
    ]));
    assert.equal(v.winRate, 0.1);
    assert.equal(v.pass, false);
  });
});

describe("opponentsFor", () => {
  it("is the rookie first, then every roster entry but the candidate", () => {
    assert.deepEqual(opponentsFor("dummy").map((e) => e.id), ["rookie"]);
    assert.deepEqual(opponentsFor(undefined).map((e) => e.id), roster.map((e) => e.id));
  });

  it("lets the rookie play itself", () => {
    assert.deepEqual(opponentsFor("rookie").map((e) => e.id), roster.map((e) => e.id));
  });
});

describe("loadCandidate", () => {
  it("takes a roster id", () => {
    assert.deepEqual(loadCandidate("dummy"), { id: "dummy", player: dummy });
  });

  it("takes a valid brain file, named after the file", () => {
    const candidate = loadCandidate(tempFile(JSON.stringify(dummyBrain)));
    assert.equal(candidate.id, undefined);
    assert.equal(candidate.player.name, "brain.json");
  });

  it("refuses an invalid brain file with its problems", () => {
    const path = tempFile(JSON.stringify({ ...dummyBrain, format: "not-a-brain", sizes: [1] }));
    assert.throws(() => loadCandidate(path), (error: Error) => {
      assert.match(error.message, /isn't a valid brain/);
      assert.match(error.message, /format/);
      return true;
    });
  });

  it("refuses a file that isn't JSON, and a name that's neither", () => {
    assert.throws(() => loadCandidate(tempFile("{nope")), /isn't JSON/);
    assert.throws(() => loadCandidate("no-such-snake"), /no roster snake or brain file/);
  });
});

describe("the gauntlet command", () => {
  it("refuses an invalid brain file with its problems, and plays nothing", () => {
    const path = tempFile(JSON.stringify({ ...dummyBrain, format: "not-a-brain" }));
    const run = cli(path, "--seeds", "1");
    assert.notEqual(run.status, 0);
    assert.match(run.stderr, /isn't a valid brain/);
    assert.doesNotMatch(run.stdout + run.stderr, /matches in|PASS|FAIL/);
  });

  it("prints the table and the verdict, failing Dummy with a non-zero exit code", () => {
    const run = cli("dummy", "--seeds", "1");
    assert.equal(run.status, 1, run.stderr);
    assert.match(run.stdout, /Rookie/);
    assert.match(run.stdout, /FAIL/);
  });
});
