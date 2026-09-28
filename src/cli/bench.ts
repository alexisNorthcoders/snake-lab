import { performance } from "node:perf_hooks";
import { parseArgs } from "node:util";
import { checkMatchOptions, playMatch } from "../match.ts";
import { exit, matchArgs, matchArgsHelp, matchOptions, orExit, parseNumber } from "./args.ts";

const usage = `Plays many matches, one seed after another, and prints ticks per second.

Usage: npm run bench -- [options]

  --matches <n>     how many matches (default 500)
${matchArgsHelp}`;

const args = orExit(usage, () => parseArgs({
  options: { ...matchArgs, matches: { type: "string", default: "500" } },
  strict: true
}).values);
if (args.help) exit(usage, 0);

const run = orExit(usage, () => {
  const matches = parseNumber("matches", args.matches);
  if (!Number.isInteger(matches) || matches < 1) throw new Error(`--matches must be a whole number above 0, not ${matches}`);
  const options = matchOptions(args);
  checkMatchOptions(options);
  return { ...options, matches };
});

let ticks = 0;
const start = performance.now();
for (let i = 0; i < run.matches; i++) {
  ticks += playMatch({ ...run, seed: (run.seed + i) % 2 ** 32, keepEvents: false }).ticks;
}
const seconds = (performance.now() - start) / 1000;

const players = run.seats.map(({ player, delay }) => `${player.name} (delay ${delay})`).join(" vs ");
console.log(`${players}, ${run.mode} at ${run.fps} fps, seeds ${run.seed} to ${run.seed + run.matches - 1}`);
console.log(`${run.matches} matches, ${ticks} ticks in ${seconds.toFixed(2)}s`);
console.log(`${Math.round(ticks / seconds)} ticks per second (${(run.matches / seconds).toFixed(1)} matches per second)`);
