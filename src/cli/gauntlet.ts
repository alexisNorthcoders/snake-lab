import { performance } from "node:perf_hooks";
import { parseArgs } from "node:util";
import { BAR_DELAY, BAR_OPPONENT, GauntletCell, loadCandidate, opponentsFor, rates, runGauntlet, verdict } from "../gauntlet.ts";
import { exit, orExit, parseNumber } from "./args.ts";

const usage = `Plays a candidate against the rookie and every other roster snake, at every delay and in both
modes, prints how often it wins, and says whether it passes the promotion bar. Exits 0 on a pass.

Usage: npm run gauntlet -- <candidate> [options]

  <candidate>       a roster id, or the path to a brain file
  --seeds <n>       seeds per opponent, delay and mode, each played in both seats (default 100)
  --base-seed <n>   the first seed, a whole number in [0, 2^32) (default 1)
  --fps <n>         ticks per second, which sets a timed round's length (default 8)`;

const parsed = orExit(usage, () => parseArgs({
  options: {
    seeds: { type: "string", default: "100" },
    "base-seed": { type: "string", default: "1" },
    fps: { type: "string", default: "8" },
    help: { type: "boolean" }
  },
  allowPositionals: true,
  strict: true
}));
if (parsed.values.help) exit(usage, 0);

const run = orExit(usage, () => {
  const { positionals, values } = parsed;
  if (positionals.length !== 1) throw new Error(`give one candidate, not ${positionals.length}`);
  const seeds = parseNumber("seeds", values.seeds);
  if (!Number.isInteger(seeds) || seeds < 1) throw new Error(`--seeds must be a whole number above 0, not ${seeds}`);
  const baseSeed = parseNumber("base-seed", values["base-seed"]);
  if (!Number.isInteger(baseSeed) || baseSeed < 0 || baseSeed >= 2 ** 32) {
    throw new Error(`--base-seed must be a whole number in [0, 2^32), not ${baseSeed}`);
  }
  const fps = parseNumber("fps", values.fps);
  if (fps <= 0) throw new Error(`--fps must be above 0, not ${fps}`);
  return { candidate: loadCandidate(positionals[0]), seeds, baseSeed, fps };
});

const opponents = opponentsFor(run.candidate.id);
const start = performance.now();
const report = runGauntlet({ candidate: run.candidate.player, opponents, baseSeed: run.baseSeed, seeds: run.seeds, fps: run.fps });
const seconds = (performance.now() - start) / 1000;

const percent = (rate: number) => `${(rate * 100).toFixed(1)}%`.padStart(6);
const nameOf = new Map(opponents.map((e) => [e.id, e.name]));
const row = (c: GauntletCell) => {
  const { win, loss, draw } = rates(c);
  return `  ${nameOf.get(c.opponent)!.padEnd(12)} ${String(c.delay).padStart(5)}  ${c.mode.padEnd(8)} ${String(c.matches).padStart(7)}  ` +
    `${percent(win)}  ${percent(loss)}  ${percent(draw)}`;
};

const matches = report.cells.reduce((sum, c) => sum + c.matches, 0);
console.log(`${report.candidate}: seeds ${report.baseSeed} to ${report.baseSeed + report.seeds - 1}, each in both seats, at ${report.fps} fps`);
console.log(`${matches} matches in ${seconds.toFixed(1)}s\n`);
console.log(`  ${"opponent".padEnd(12)} delay  ${"mode".padEnd(8)} matches     win    loss    draw`);
report.cells.forEach((c) => console.log(row(c)));

const { bar, winRate, pass } = verdict(report);
console.log(`\nBar: win at least ${percent(bar).trim()} against ${nameOf.get(BAR_OPPONENT)} at delay ${BAR_DELAY}, both modes together.`);
console.log(`${report.candidate} won ${percent(winRate).trim()}: ${pass ? "PASS" : "FAIL"}`);
process.exit(pass ? 0 : 1);
