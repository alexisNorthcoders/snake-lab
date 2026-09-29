import assert from "node:assert/strict";
import { appendFileSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { get } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";
import { LogReader } from "../src/dashboard/log.ts";
import { startDashboard } from "../src/dashboard/server.ts";
import { createRun, runFiles } from "../src/run.ts";
import { DEFAULT_SETTINGS, GenerationLog, TrainSettings, generationRng, randomBrain } from "../src/train.ts";

const settings: TrainSettings = { ...DEFAULT_SETTINGS, personality: "glutton", seed: 3, generations: 6, aloneGenerations: 2, rookieGenerations: 2, population: 4, matches: 1, hidden: [4] };
const line = (generation: number, best = generation * 2): GenerationLog => ({ generation, stage: generation < 2 ? "alone" : "rookie", best, mean: best / 2 });
const text = (...lines: GenerationLog[]) => lines.map((l) => `${JSON.stringify(l)}\n`).join("");

const newRun = () => {
  const run = join(mkdtempSync(join(tmpdir(), "dash-")), "run");
  createRun(run, settings);
  return run;
};

/** Every file in the folder with its size and mtime, to see nothing changed. */
const snapshot = (run: string) => readdirSync(run, { recursive: true, withFileTypes: true })
  .map((e) => join(e.parentPath, e.name)).sort().map((p) => `${p} ${statSync(p).isFile() ? readFileSync(p, "utf8").length : "dir"} ${statSync(p).mtimeMs}`);

/** Connects to /events and collects `event` names and data as they arrive. */
function listen(url: string) {
  const events: { event: string; data: any }[] = [];
  const waiters: (() => void)[] = [];
  const req = get(`${url}/events`, (res) => {
    let buffer = "";
    res.setEncoding("utf8");
    res.on("data", (chunk: string) => {
      buffer += chunk;
      let end;
      while ((end = buffer.indexOf("\n\n")) >= 0) {
        const block = buffer.slice(0, end);
        buffer = buffer.slice(end + 2);
        const event = /^event: (.*)$/m.exec(block)?.[1];
        const data = /^data: (.*)$/m.exec(block)?.[1];
        if (event && data) events.push({ event, data: JSON.parse(data) });
        waiters.splice(0).forEach((w) => w());
      }
    });
  });
  req.on("error", () => {});
  return {
    events,
    async until(count: number) {
      while (events.length < count) await new Promise<void>((done) => { waiters.push(done); setTimeout(done, 2000); });
      assert.ok(events.length >= count, `waited for ${count} events, got ${events.length}`);
    },
    close: () => req.destroy()
  };
}

describe("LogReader", () => {
  it("reads every generation of a finished run", () => {
    const run = newRun();
    writeFileSync(runFiles(run).log, text(line(0), line(1), line(2)));
    const reader = new LogReader(runFiles(run).log);
    assert.deepEqual(reader.poll(), { kind: "append", lines: [line(0), line(1), line(2)] });
    assert.deepEqual(reader.all, [line(0), line(1), line(2)]);
    assert.equal(reader.poll(), undefined);
  });

  it("returns new lines as they're appended, and waits for a line to be whole", () => {
    const run = newRun();
    const path = runFiles(run).log;
    const reader = new LogReader(path);
    assert.equal(reader.poll(), undefined); // no log file yet
    writeFileSync(path, "");
    assert.equal(reader.poll(), undefined); // an empty log
    assert.deepEqual(reader.all, []);
    appendFileSync(path, text(line(0)));
    assert.deepEqual(reader.poll(), { kind: "append", lines: [line(0)] });
    appendFileSync(path, JSON.stringify(line(1)).slice(0, 10));
    assert.equal(reader.poll(), undefined);
    appendFileSync(path, `${JSON.stringify(line(1)).slice(10)}\n`);
    assert.deepEqual(reader.poll(), { kind: "append", lines: [line(1)] });
  });

  it("gives the whole log again when a resume drops and rewrites lines", () => {
    const run = newRun();
    const path = runFiles(run).log;
    writeFileSync(path, text(line(0), line(1), line(2)));
    const reader = new LogReader(path);
    reader.poll();
    writeFileSync(path, text(line(0), line(1)));
    assert.deepEqual(reader.poll(), { kind: "reset", lines: [line(0), line(1)] });
    writeFileSync(path, text(line(0), line(1), line(2, 99)));
    assert.deepEqual(reader.poll(), { kind: "append", lines: [line(2, 99)] });
    writeFileSync(path, text(line(0), line(1, 50), line(2, 99)));
    assert.equal(reader.poll()?.kind, "reset");
  });
});

describe("dashboard", () => {
  const serve = async (run: string) => startDashboard(run, { port: 0, pollMs: 20 });

  it("refuses a folder that isn't a run", async () => {
    const empty = mkdtempSync(join(tmpdir(), "dash-"));
    await assert.rejects(startDashboard(empty, { port: 0 }), /isn't a run folder/);
    await assert.rejects(startDashboard(join(empty, "missing"), { port: 0 }), /isn't a run folder/);
    const bad = join(empty, "bad");
    mkdirSync(bad);
    writeFileSync(join(bad, "settings.json"), "nope");
    await assert.rejects(startDashboard(bad, { port: 0 }), /isn't a run folder/);
  });

  it("serves the page and the run's settings", async () => {
    const run = newRun();
    const dashboard = await serve(run);
    try {
      const page = await fetch(`${dashboard.url}/`);
      assert.match(await page.text(), /<svg id="chart"/);
      const info = await (await fetch(`${dashboard.url}/api/run`)).json();
      assert.equal(info.settings.personality, "glutton");
      assert.equal((await fetch(`${dashboard.url}/nothing`)).status, 404);
    } finally {
      await dashboard.close();
    }
  });

  it("sends the whole log on connect, then one event per new line", async () => {
    const run = newRun();
    const path = runFiles(run).log;
    writeFileSync(path, text(line(0), line(1)));
    const dashboard = await serve(run);
    const client = listen(dashboard.url);
    try {
      await client.until(1);
      assert.equal(client.events[0].event, "log");
      assert.deepEqual(client.events[0].data.lines, [line(0), line(1)]);
      appendFileSync(path, text(line(2)));
      await client.until(2);
      appendFileSync(path, text(line(3), line(4)));
      await client.until(4);
      assert.deepEqual(client.events.slice(1).map((e) => [e.event, e.data.line]), [["generation", line(2)], ["generation", line(3)], ["generation", line(4)]]);
      writeFileSync(path, text(line(0), line(1), line(2)));
      await client.until(5);
      assert.equal(client.events[4].event, "log");
      assert.deepEqual(client.events[4].data.lines, [line(0), line(1), line(2)]);
    } finally {
      client.close();
      await dashboard.close();
    }
  });

  it("starts on a run with no log lines yet", async () => {
    const run = newRun();
    const dashboard = await serve(run);
    const client = listen(dashboard.url);
    try {
      await client.until(1);
      assert.deepEqual(client.events[0].data.lines, []);
      appendFileSync(runFiles(run).log, text(line(0)));
      await client.until(2);
      assert.deepEqual(client.events[1].data.line, line(0));
    } finally {
      client.close();
      await dashboard.close();
    }
  });

  it("streams a generation's sample games tick by tick, and refuses one that isn't saved", async () => {
    const run = newRun();
    writeFileSync(runFiles(run).generation(0), JSON.stringify(randomBrain(generationRng(1, 0), settings.hidden, settings.activation, 0.5)));
    const dashboard = await startDashboard(run, { port: 0, games: 2 });
    try {
      assert.equal((await fetch(`${dashboard.url}/games?generation=1`)).status, 404);
      assert.equal((await fetch(`${dashboard.url}/games?generation=x`)).status, 404);
      const body = await (await fetch(`${dashboard.url}/games?generation=0`)).text();
      const events = [...body.matchAll(/^event: (.*)\ndata: (.*)$/gm)].map((m) => ({ event: m[1], data: JSON.parse(m[2]) }));
      assert.deepEqual({ ...events[0].data, network: undefined }, { generation: 0, games: 1, network: undefined });
      assert.deepEqual(events[0].data.network.columns.map((c: { labels: string[] }) => c.labels.length), [23, ...settings.hidden, 3]);
      assert.ok(events.filter((e) => e.event === "tick").some((e) => e.data.activations?.values.length === settings.hidden.length + 2));
      assert.equal(events[events.length - 1].event, "done");
      assert.deepEqual(events.filter((e) => e.event === "start").map((e) => e.data.game.index), [0]);
      assert.equal(events.filter((e) => e.event === "end").length, 1);
      assert.ok(events.filter((e) => e.event === "tick").length > 2);
      const info = await (await fetch(`${dashboard.url}/api/run`)).json();
      assert.deepEqual(info.replay, { speed: 8, games: 2 });
    } finally {
      await dashboard.close();
    }
  });

  it("never writes into the run folder", async () => {
    const run = newRun();
    writeFileSync(runFiles(run).log, text(line(0), line(1)));
    const before = snapshot(run);
    const dashboard = await serve(run);
    const client = listen(dashboard.url);
    try {
      await client.until(1);
      await fetch(`${dashboard.url}/`);
      await fetch(`${dashboard.url}/api/run`);
      await fetch(`${dashboard.url}/events`, { method: "POST", body: "x" });
      await new Promise((done) => setTimeout(done, 100));
      assert.deepEqual(snapshot(run), before);
    } finally {
      client.close();
      await dashboard.close();
    }
  });
});
