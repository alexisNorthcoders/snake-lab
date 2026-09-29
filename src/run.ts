import { closeSync, existsSync, fsyncSync, mkdirSync, openSync, readFileSync, renameSync, writeSync } from "node:fs";
import { join } from "node:path";
import { Brain } from "snake-colyseus/bots";
import { Checkpoint, DEFAULT_SETTINGS, GenerationLog, TrainResult, TrainSettings, checkSettings, stageOf, train } from "./train.ts";

/**
 * A run folder:
 *
 * - `settings.json`: the run's settings
 * - `log.jsonl`: one `GenerationLog` line a generation
 * - `generations/gen-NNNN.json`: each generation's fittest, a brain file
 * - `best.json`: the latest generation's fittest, the same brain again
 * - `checkpoint.json`: the generation to play next and its whole population, the only one on disk
 *
 * Every file but the log is written to a `.tmp` file and renamed into place,
 * and the checkpoint last of a generation's files: a crash mid-write leaves the
 * last whole checkpoint, and the log lines past it are dropped on resume.
 */
export const runFiles = (run: string) => ({
  settings: join(run, "settings.json"),
  log: join(run, "log.jsonl"),
  checkpoint: join(run, "checkpoint.json"),
  best: join(run, "best.json"),
  generations: join(run, "generations"),
  generation: (g: number) => join(run, "generations", `gen-${String(g).padStart(4, "0")}.json`)
});

/** Writes `text` so that `path` is always either its old contents or all of `text`. */
function writeWhole(path: string, text: string) {
  const tmp = `${path}.tmp`;
  const fd = openSync(tmp, "w");
  try {
    writeSync(fd, text);
    fsyncSync(fd);
  } finally {
    closeSync(fd);
  }
  renameSync(tmp, path);
}

const json = (value: unknown) => `${JSON.stringify(value, null, 2)}\n`;

/** Makes a new run folder with its settings. Throws if the folder already exists. */
export function createRun(run: string, settings: TrainSettings) {
  checkSettings(settings);
  if (existsSync(run)) throw new Error(`${run} already exists: name a new run folder, or --resume it`);
  const files = runFiles(run);
  mkdirSync(files.generations, { recursive: true });
  writeWhole(files.settings, json(settings));
}

/** Reads a checkpoint, refusing one that isn't whole: wrong shape, or not `population` brains of the run's size. */
function readCheckpoint(path: string, settings: TrainSettings): Checkpoint {
  let checkpoint: Checkpoint;
  try {
    checkpoint = JSON.parse(readFileSync(path, "utf8"));
  } catch (error) {
    throw new Error(`${path} isn't whole JSON (${(error as Error).message}): the run can't be resumed from it`);
  }
  const { generation, population } = checkpoint ?? ({} as Partial<Checkpoint>);
  // A checkpoint from before the league has no stage: it's the one its generation is in.
  if (checkpoint && Number.isInteger(generation) && checkpoint.stage === undefined) checkpoint.stage = stageOf(generation, settings);
  if (!Number.isInteger(generation) || generation < 0 || !Array.isArray(population) || population.length !== settings.population ||
    population.some((brain) => !brain || typeof brain !== "object" || !Array.isArray((brain as { layers?: unknown }).layers))) {
    throw new Error(`${path} is malformed: it needs a generation and ${settings.population} brains`);
  }
  return checkpoint;
}

/** Saved settings as a run reads them: a run from before the league stays in the rookie stage to its end, and gets the defaults of the settings it lacks. */
export const withDefaults = (saved: Partial<TrainSettings> & { generations: number; aloneGenerations: number }): TrainSettings =>
  ({ ...DEFAULT_SETTINGS, rookieGenerations: saved.generations - saved.aloneGenerations, ...saved }) as TrainSettings;

/** A run folder's settings, and its latest checkpoint if it has one. Drops log lines past the checkpoint. */
export function openRun(run: string): { settings: TrainSettings; from?: Checkpoint; log: GenerationLog[] } {
  const files = runFiles(run);
  if (!existsSync(files.settings)) throw new Error(`${run} isn't a run folder: it has no settings.json`);
  const saved = JSON.parse(readFileSync(files.settings, "utf8"));
  const settings = withDefaults(saved);
  const from: Checkpoint | undefined = existsSync(files.checkpoint) ? readCheckpoint(files.checkpoint, settings) : undefined;
  const played = from?.generation ?? 0;
  const lines = (existsSync(files.log) ? readFileSync(files.log, "utf8") : "").split("\n");
  if (lines.length - 1 < played) throw new Error(`${files.log} has fewer lines than the checkpoint's ${played} generations`);
  const log: GenerationLog[] = lines.slice(0, played).map((line) => JSON.parse(line));
  writeWhole(files.log, log.map((line) => `${JSON.stringify(line)}\n`).join(""));
  return { settings, from, log };
}

/** Saves a generation: its fittest, its log line, then the checkpoint to carry on from. */
export function saveGeneration(run: string, line: GenerationLog, best: Brain, next: Checkpoint) {
  const files = runFiles(run);
  writeWhole(files.generation(line.generation), json(best));
  writeWhole(files.best, json(best));
  const fd = openSync(files.log, "a");
  try {
    writeSync(fd, `${JSON.stringify(line)}\n`);
    fsyncSync(fd);
  } finally {
    closeSync(fd);
  }
  writeWhole(files.checkpoint, JSON.stringify(next));
}

export interface RunOptions {
  workers?: number;
  /** A new target generation count, saved to the run's settings. The rest of them can't change. */
  generations?: number;
  onGeneration?: (line: GenerationLog) => void;
  stop?: () => boolean;
}

/**
 * Trains a run folder made by `createRun`: from generation 0, or from its
 * latest checkpoint, saving every generation. Undefined when the run already has
 * all its generations. The same as a run that never stopped.
 */
export async function trainRun(run: string, options: RunOptions = {}): Promise<TrainResult | undefined> {
  const opened = openRun(run);
  let { settings } = opened;
  const played = opened.from?.generation ?? 0;
  if (options.generations !== undefined && options.generations !== settings.generations) {
    if (options.generations < played) {
      throw new Error(`${run} has already played ${played} generations: can't cut it to ${options.generations}`);
    }
    settings = { ...settings, generations: options.generations };
    checkSettings(settings);
    writeWhole(runFiles(run).settings, json(settings));
  }
  if (played >= settings.generations) return undefined;
  return train(settings, {
    workers: options.workers,
    from: opened.from,
    stop: options.stop,
    onGeneration: (line, best, next) => {
      saveGeneration(run, line, best, next);
      options.onGeneration?.(line);
    }
  });
}
