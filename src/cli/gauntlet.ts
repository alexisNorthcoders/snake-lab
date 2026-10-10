import { performance } from "node:perf_hooks";
import { parseArgs } from "node:util";
import { formatReport, loadCandidate, opponentsFor, runGauntlet, verdict } from "../gauntlet.ts";
import { exit, gauntletArgs, gauntletArgsHelp, gauntletSettings, orExit } from "./args.ts";

const usage = `Plays a candidate against the rookie and every other roster snake, at every delay and in both
modes, prints how often it wins, and says whether it passes the promotion bar. Exits 0 on a pass.

Usage: npm run gauntlet -- <candidate> [options]

  <candidate>       a roster id, or the path to a brain file
${gauntletArgsHelp}`;

const parsed = orExit(usage, () => parseArgs({
  options: {
    ...gauntletArgs,
    help: { type: "boolean" }
  },
  allowPositionals: true,
  strict: true
}));
if (parsed.values.help) exit(usage, 0);

const run = orExit(usage, () => {
  const { positionals, values } = parsed;
  if (positionals.length !== 1) throw new Error(`give one candidate, not ${positionals.length}`);
  return { candidate: loadCandidate(positionals[0]), ...gauntletSettings(values) };
});

const opponents = opponentsFor(run.candidate.id);
const start = performance.now();
const report = runGauntlet({ candidate: run.candidate.player, opponents, baseSeed: run.baseSeed, seeds: run.seeds, fps: run.fps });
const seconds = (performance.now() - start) / 1000;

console.log(formatReport(report, opponents, seconds, run.bar));
process.exit(verdict(report, run.bar).pass ? 0 : 1);
