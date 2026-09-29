import { existsSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { Brain, brainDecider, brainProblems, loadRoster, roster } from "snake-colyseus/bots";
import { GauntletOptions, GauntletReport, Verdict, formatReport, opponentsFor, runGauntlet, verdict } from "./gauntlet.ts";
import { runFiles } from "./run.ts";
import { TrainSettings } from "./train.ts";

/** Where the snake-colyseus checkout is, unless told otherwise. */
export const DEFAULT_TARGET = "../snake-colyseus";

export interface PromoteOptions {
  /** The run folder. */
  run: string;
  /** Which generation's fittest to put forward. */
  generation: number;
  /** The snake's name in game; its id is a slug of it. */
  name: string;
  /** The snake-colyseus checkout to write into (default `DEFAULT_TARGET`). */
  target?: string;
  seeds: number;
  baseSeed: number;
  fps: number;
  /** Plays the gauntlet: `runGauntlet`, unless a test stands in for it. */
  gauntlet?: (options: GauntletOptions) => GauntletReport;
}

export interface PromoteResult {
  id: string;
  report: GauntletReport;
  verdict: Verdict;
  /** The gauntlet's table and verdict, as `gauntlet` prints them. */
  table: string;
  /** On a pass: the PR description, and every file written. */
  promotion?: { description: string; written: string[] };
}

/** A roster id from a name: lower case letters and digits, the rest becoming single hyphens. */
export const slug = (name: string) =>
  name.normalize("NFKD").replace(/[̀-ͯ]/g, "").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");

/** `version` with its minor part up by one and its patch reset. Throws on anything but `x.y.z`. */
export function bumpMinor(version: string): string {
  const match = /^(\d+)\.(\d+)\.\d+$/.exec(version);
  if (!match) throw new Error(`engineVersion "${version}" isn't x.y.z: can't bump it`);
  return `${match[1]}.${Number(match[2]) + 1}.0`;
}

/** The engine tag the lab is installed at, read from the installed package. */
function engineTag(): string {
  const path = fileURLToPath(new URL("../node_modules/snake-colyseus/package.json", import.meta.url));
  const { engineVersion } = JSON.parse(readFileSync(path, "utf8")) as { engineVersion?: string };
  if (!engineVersion) throw new Error(`${path} has no engineVersion: reinstall snake-colyseus`);
  return `engine-v${engineVersion}`;
}

/** The checkpoint's brain, or why it can't be used. */
function readCheckpoint(path: string): Brain {
  if (!existsSync(path)) throw new Error(`no checkpoint at ${path}: the run hasn't played that generation`);
  let brain: unknown;
  try {
    brain = JSON.parse(readFileSync(path, "utf8"));
  } catch (error) {
    throw new Error(`${path} isn't JSON: ${(error as Error).message}`);
  }
  const problems = brainProblems(brain);
  if (problems.length > 0) throw new Error(`${path} isn't a valid brain:\n${problems.map((p) => `  - ${p}`).join("\n")}`);
  return brain as Brain;
}

/** The pieces of a snake-colyseus checkout `promote` touches, checked. */
function openTarget(target: string) {
  const packagePath = join(target, "package.json");
  const bots = join(target, "src", "bots");
  let pkg: { name?: string; engineVersion?: string };
  try {
    pkg = JSON.parse(readFileSync(packagePath, "utf8"));
  } catch {
    throw new Error(`${target} isn't a snake-colyseus checkout: it has no readable package.json`);
  }
  if (pkg.name !== "snake-colyseus" || !existsSync(join(bots, "entries")) || !existsSync(join(bots, "brains"))) {
    throw new Error(`${target} isn't a snake-colyseus checkout: it needs package.json named snake-colyseus and src/bots/{entries,brains}`);
  }
  if (typeof pkg.engineVersion !== "string") throw new Error(`${packagePath} has no engineVersion`);
  return { packagePath, bots, engineVersion: pkg.engineVersion };
}

/** The command that reproduces the gauntlet's numbers. */
const commandFor = (o: PromoteOptions) =>
  `npm run promote -- ${o.run} ${o.generation} --name ${JSON.stringify(o.name)} --seeds ${o.seeds} --base-seed ${o.baseSeed} --fps ${o.fps}`;

function describePromotion(o: PromoteOptions, id: string, settings: TrainSettings, table: string, engine: string, shipsAs: string): string {
  return [
    `# Add ${o.name} to the roster`,
    "",
    `- **Name:** ${o.name} (\`${id}\`)`,
    `- **Personality:** ${settings.personality}`,
    `- **Generation:** ${o.generation}, of run \`${o.run}\` (neuroevolution, seed ${settings.seed})`,
    `- **Engine:** played under \`${engine}\`; this PR bumps the target's \`engineVersion\`, so it will be tagged \`${shipsAs}\``,
    "",
    "## Training settings",
    "",
    "```json",
    JSON.stringify(settings, null, 2),
    "```",
    "",
    "## Gauntlet",
    "",
    "```sh",
    commandFor(o),
    "```",
    "",
    "```",
    table,
    "```",
    ""
  ].join("\n");
}

/**
 * Puts a generation's fittest through the gauntlet and, if it passes, adds it
 * to a snake-colyseus checkout: its brain, its roster entry, and a minor bump
 * of `engineVersion`. Throws, writing nothing, when the id or name is taken,
 * the target isn't a checkout or the checkpoint can't be used. On a fail it
 * writes nothing either. It never runs git.
 */
export function promote(options: PromoteOptions): PromoteResult {
  const { run, generation, name, target = DEFAULT_TARGET, gauntlet = runGauntlet } = options;
  const id = slug(name);
  if (id === "") throw new Error(`"${name}" has no letters or digits to make an id from`);
  if (!Number.isInteger(generation) || generation < 0) throw new Error(`the generation must be a whole number, 0 or more, not ${generation}`);

  const files = runFiles(run);
  if (!existsSync(files.settings)) throw new Error(`${run} isn't a run folder: it has no settings.json`);
  const settings: TrainSettings = JSON.parse(readFileSync(files.settings, "utf8"));
  const brain = readCheckpoint(files.generation(generation));

  const checkout = openTarget(target);
  const taken = loadRoster(checkout.bots, { error: () => {}, warn: () => {} });
  const entryPath = join(checkout.bots, "entries", `${id}.json`);
  const brainPath = join(checkout.bots, "brains", `${id}.json`);
  if (taken.some((e) => e.id === id) || existsSync(entryPath) || existsSync(brainPath)) {
    throw new Error(`the id "${id}" is taken in ${target}'s roster: pick another name`);
  }
  if (taken.some((e) => e.name.toLowerCase() === name.toLowerCase())) {
    throw new Error(`the name "${name}" is taken in ${target}'s roster: pick another`);
  }
  const nextVersion = bumpMinor(checkout.engineVersion);
  const packageText = readFileSync(checkout.packagePath, "utf8");
  const bumped = packageText.replace(/("engineVersion"\s*:\s*")[^"]*(")/, `$1${nextVersion}$2`);
  if (bumped === packageText) throw new Error(`couldn't find engineVersion in ${checkout.packagePath} to bump`);

  const opponents = opponentsFor(undefined, roster);
  const report = gauntlet({
    candidate: { name, decider: brainDecider(brain) },
    opponents,
    baseSeed: options.baseSeed,
    seeds: options.seeds,
    fps: options.fps
  });
  const table = formatReport(report, opponents);
  const result = verdict(report);
  if (!result.pass) return { id, report, verdict: result, table };

  const entry = { id, name, personality: settings.personality, generation, method: "neuroevolution", brain: `${id}.json` };
  const description = describePromotion(options, id, settings, table, engineTag(), `engine-v${nextVersion}`);
  const descriptionPath = join(run, `promote-${id}.md`);
  const descriptionExisted = existsSync(descriptionPath);
  const previousDescription = descriptionExisted ? readFileSync(descriptionPath, "utf8") : "";
  const outputs: [string, string][] = [
    [brainPath, `${JSON.stringify(brain, null, 2)}\n`],
    [entryPath, `${JSON.stringify(entry, null, 2)}\n`],
    [descriptionPath, description],
    [checkout.packagePath, bumped]
  ];
  // Stage every file first, then rename them into place; a failure at either step undoes what was done.
  const renamed: string[] = [];
  try {
    for (const [path, text] of outputs) writeFileSync(`${path}.tmp`, text);
    for (const [path] of outputs) {
      renameSync(`${path}.tmp`, path);
      renamed.push(path);
    }
  } catch (error) {
    for (const [path] of outputs) rmSync(`${path}.tmp`, { force: true });
    for (const path of renamed) {
      if (path === checkout.packagePath) writeFileSync(path, packageText);
      else if (path === descriptionPath && descriptionExisted) writeFileSync(path, previousDescription);
      else rmSync(path, { force: true });
    }
    throw error;
  }
  const written = [brainPath, entryPath, checkout.packagePath, descriptionPath].map((p) => resolve(p));
  return { id, report, verdict: result, table, promotion: { description, written } };
}
