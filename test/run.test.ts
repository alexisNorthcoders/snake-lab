import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import { appendFileSync, mkdtempSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, relative } from "node:path";
import { describe, it } from "node:test";
import { loadCandidate } from "../src/gauntlet.ts";
import { createRun, runFiles, trainRun } from "../src/run.ts";
import { DEFAULT_SETTINGS, TrainSettings } from "../src/train.ts";

const short: TrainSettings = {
  ...DEFAULT_SETTINGS, personality: "glutton", seed: 3, generations: 4, aloneGenerations: 1, population: 6, matches: 2, hidden: [4]
};

const newRun = () => join(mkdtempSync(join(tmpdir(), "run-")), "run");

/** Every file in the folder, by its path in the folder. */
const contents = (run: string): Record<string, string> => Object.fromEntries(readdirSync(run, { recursive: true, withFileTypes: true })
  .filter((entry) => entry.isFile())
  .map((entry) => [relative(run, join(entry.parentPath, entry.name)), readFileSync(join(entry.parentPath, entry.name), "utf8")])
  .sort(([a], [b]) => a.localeCompare(b)));

/** A run straight through, and its folder. */
async function straight(settings = short) {
  const run = newRun();
  createRun(run, settings);
  await trainRun(run);
  return contents(run);
}

/** The first `k` generations of a run, stopped there. */
async function stoppedAfter(k: number, settings = short) {
  const run = newRun();
  createRun(run, settings);
  let played = 0;
  await trainRun(run, { onGeneration: () => played++, stop: () => played === k });
  return run;
}

describe("a run folder", () => {
  it("keeps every generation's fittest as a brain file, and one population: the checkpoint's", async () => {
    const files = await straight();
    assert.deepEqual(Object.keys(files), [
      "best.json", "checkpoint.json", "generations/gen-0000.json", "generations/gen-0001.json",
      "generations/gen-0002.json", "generations/gen-0003.json", "log.jsonl", "settings.json"
    ]);
    const populations = Object.entries(files).filter(([, text]) => text.includes('"population":['));
    assert.deepEqual(populations.map(([name]) => name), ["checkpoint.json"]);
    const checkpoint = JSON.parse(files["checkpoint.json"]);
    assert.equal(checkpoint.generation, 4);
    assert.equal(checkpoint.population.length, short.population);
    assert.equal(files["log.jsonl"].trim().split("\n").length, 4);
    assert.equal(files["best.json"], files["generations/gen-0003.json"]);
  });

  it("saves brain files that pass the brain-file check", async () => {
    const run = await stoppedAfter(2);
    for (const g of [0, 1]) assert.doesNotThrow(() => loadCandidate(runFiles(run).generation(g)));
  });

  it("resumed after generation k, gives the same logs and brains as a run straight through", async () => {
    const expected = await straight();
    for (const k of [1, 3]) {
      const run = await stoppedAfter(k);
      assert.equal(readFileSync(runFiles(run).log, "utf8").trim().split("\n").length, k);
      await trainRun(run);
      assert.deepEqual(contents(run), expected);
    }
  });

  it("never resumes from a half-written checkpoint, nor keeps a half-written generation's log line", async () => {
    const expected = await straight();
    const run = await stoppedAfter(2);
    const files = runFiles(run);
    // A crash writing generation 2: its brain file and log line are out, its checkpoint half-written.
    writeFileSync(files.generation(2), '{"format":');
    appendFileSync(files.log, '{"generation":2,"stage":"rookie","best":1');
    writeFileSync(`${files.checkpoint}.tmp`, readFileSync(files.checkpoint, "utf8").slice(0, 100));
    await trainRun(run);
    const resumed = contents(run);
    delete resumed["checkpoint.json.tmp"];
    assert.deepEqual(resumed, expected);
  });

  it("carries on to more generations, the only setting besides workers that can change", async () => {
    const expected = await straight();
    const run = newRun();
    createRun(run, { ...short, generations: 2 });
    await trainRun(run);
    assert.equal(await trainRun(run), undefined, "it has all its generations");
    await trainRun(run, { generations: 4 });
    assert.deepEqual(contents(run), expected);
    await assert.rejects(trainRun(run, { generations: 3 }), /already played 4 generations/);
  });

  it("gives the same folder on 1 worker and on several", async () => {
    const expected = await straight();
    const run = newRun();
    createRun(run, short);
    await trainRun(run, { workers: 3 });
    assert.deepEqual(contents(run), expected);
  });
});

describe("a run through the stages", () => {
  const staged = { ...short, personality: "hunter" as const, generations: 4, aloneGenerations: 0, rookieGenerations: 2, fourPlayerShare: 0.5 };

  it("stopped in the rookie stage and resumed into the league, matches a run straight through", async () => {
    const expected = await straight(staged);
    const run = await stoppedAfter(2, staged);
    assert.equal(JSON.parse(readFileSync(runFiles(run).checkpoint, "utf8")).stage, "league");
    await trainRun(run, { workers: 2 });
    assert.deepEqual(contents(run), expected);
    assert.deepEqual(expected["log.jsonl"].trim().split("\n").map((l) => JSON.parse(l).stage), ["rookie", "rookie", "league", "league"]);
  });

  it("puts the stage in the checkpoint, and resumes an old checkpoint without one", async () => {
    const run = await stoppedAfter(1, staged);
    const files = runFiles(run);
    const checkpoint = JSON.parse(readFileSync(files.checkpoint, "utf8"));
    assert.equal(checkpoint.stage, "rookie");
    delete checkpoint.stage;
    writeFileSync(files.checkpoint, JSON.stringify(checkpoint));
    await trainRun(run);
    assert.deepEqual(contents(run), await straight(staged));
  });
});

describe("train --resume", () => {
  const args = ["--personality", "glutton", "--seed", "3", "--generations", "20", "--alone", "1", "--population", "6", "--matches", "2", "--hidden", "4"];
  const cli = (...more: string[]) => spawnSync(process.execPath, ["--import", "tsx", "src/cli/train.ts", ...more], { encoding: "utf8" });

  it("stops after the current generation on SIGTERM, then carries on to the same result", async () => {
    const expected = await straight();
    const run = newRun();
    const child = spawn(process.execPath, ["--import", "tsx", "src/cli/train.ts", ...args, "--run", run, "--workers", "2"]);
    let stdout = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
      if (stdout.includes('"generation":0') && !child.killed) child.kill("SIGTERM");
    });
    const code = await new Promise((resolve) => child.on("close", resolve));
    assert.equal(code, 0);
    const last = Number(/Stopped: generations 0 to (\d+) of 20/.exec(stdout)?.[1]);
    assert.ok(last < 2, stdout);
    assert.equal(JSON.parse(readFileSync(runFiles(run).checkpoint, "utf8")).generation, last + 1);

    const refused = cli("--resume", run, "--population", "7");
    assert.equal(refused.status, 1);
    assert.match(refused.stderr, /only --generations and --workers can be given, not --population/);

    const resumed = cli("--resume", run, "--generations", "4", "--workers", "1");
    assert.equal(resumed.status, 0, resumed.stderr);
    assert.match(resumed.stdout, /Trained: generations [12] to 3 of 4 in .* on 1 worker: .* matches a second, \d+ generations an hour/);
    const { "settings.json": settings, ...files } = contents(run);
    const { "settings.json": expectedSettings, ...expectedFiles } = expected;
    assert.deepEqual(JSON.parse(settings), JSON.parse(expectedSettings));
    assert.deepEqual(files, expectedFiles);
  });
});
