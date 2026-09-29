import { existsSync, readFileSync, statSync } from "node:fs";
import { GenerationLog } from "../train.ts";

/** What a read found since the last one. */
export type LogChange =
  /** Lines were added at the end. */
  | { kind: "append"; lines: GenerationLog[] }
  /** The log isn't what it was plus more (a resume dropped and rewrote lines, or it emptied): here it is whole. */
  | { kind: "reset"; lines: GenerationLog[] };

/** The whole lines of a log's text: a last line without its newline is still being written, so it's left out. */
export function parseLog(text: string): GenerationLog[] {
  const end = text.lastIndexOf("\n");
  if (end < 0) return [];
  return text.slice(0, end).split("\n").filter((line) => line.trim() !== "").map((line) => JSON.parse(line));
}

/** Reads a run's `log.jsonl` (never writes it) and reports what changed since the last `poll`. */
export class LogReader {
  private lines: GenerationLog[] = [];
  private text = "";

  constructor(private readonly path: string) {}

  /** Every generation logged as of the last `poll`. */
  get all(): GenerationLog[] {
    return this.lines;
  }

  /** When the log was last written, in ms since the epoch; undefined if it doesn't exist. */
  get updatedAt(): number | undefined {
    try {
      return statSync(this.path).mtimeMs;
    } catch {
      return undefined;
    }
  }

  /** Rereads the log. Undefined when nothing changed. */
  poll(): LogChange | undefined {
    const text = existsSync(this.path) ? readFileSync(this.path, "utf8") : "";
    if (text === this.text) return undefined;
    let next: GenerationLog[];
    try {
      next = parseLog(text);
    } catch {
      return undefined; // caught mid-write: try again next poll
    }
    const before = this.lines;
    this.text = text;
    this.lines = next;
    const extended = next.length >= before.length && before.every((line, i) => JSON.stringify(line) === JSON.stringify(next[i]));
    if (extended) return next.length === before.length ? undefined : { kind: "append", lines: next.slice(before.length) };
    return { kind: "reset", lines: next };
  }
}
