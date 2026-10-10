import { parseArgs } from "node:util";
import { DEFAULT_TARGET, promote } from "../promote.ts";
import { exit, gauntletArgs, gauntletArgsHelp, gauntletSettings, orExit, parseNumber } from "./args.ts";

const usage = `Puts a generation of a training run through the gauntlet and, if it passes, writes its brain and
roster entry into a snake-colyseus checkout, bumps that checkout's engineVersion by a patch version,
and prints a PR description (saved in the run folder as promote-<id>.md). If it fails, or the id or
name is taken, or the target or the checkpoint can't be used, nothing is written and it exits 1.
It never runs git: branch, commit and open the PR yourself.

Usage: npm run promote -- <run> <generation> --name <Name> [options]

  <run>             the run folder
  <generation>      the generation whose fittest snake to promote
  --name <Name>     the snake's name in game; its roster id is a slug of it
  --target <dir>    the snake-colyseus checkout to write into (default ${DEFAULT_TARGET})
${gauntletArgsHelp}`;

const parsed = orExit(usage, () => parseArgs({
  options: { name: { type: "string" }, target: { type: "string", default: DEFAULT_TARGET }, ...gauntletArgs, help: { type: "boolean" } },
  allowPositionals: true,
  strict: true
}));
if (parsed.values.help) exit(usage, 0);

const options = orExit(usage, () => {
  const { positionals, values } = parsed;
  if (positionals.length !== 2) throw new Error(`give a run folder and a generation, not ${positionals.length} arguments`);
  if (!values.name) throw new Error("--name is required");
  const generation = parseNumber("generation", positionals[1]);
  return { run: positionals[0], generation, name: values.name, target: values.target, ...gauntletSettings(values) };
});

let result;
try {
  result = promote(options);
} catch (error) {
  exit(`Refused, nothing written: ${(error as Error).message}`);
}

console.log(result.table);
if (!result.promotion) exit(`\nNot promoted: ${options.name} didn't clear the bar. Nothing written.`);
console.log(`\nWrote:\n${result.promotion.written.map((p) => `  ${p}`).join("\n")}`);
console.log(`\nNext: in ${options.target}, branch, commit these files and open a PR with this description:\n`);
console.log(result.promotion.description);
