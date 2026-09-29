import { GameMode, mulberry32 } from "snake-colyseus/engine";
import { Action, EncoderVersion, OBSERVATION_SIZE, Seat, StepFacts, SteppedMatch } from "./match.ts";
import { loadCandidate } from "./gauntlet.ts";

/** The wire protocol's version, sent in every JSON header so a reader can refuse one it doesn't know. */
export const PROTOCOL_VERSION = 1;

/** Facts a match's record carries, in order, each an int32 (see docs/env-protocol.md). */
export const FACT_FIELDS = [
  "scoreGained", "ate", "kills", "alive", "deathCause", "deathBy", "ended", "outcome", "reason", "ticks", "seat"
] as const;
export const FACT_COUNT = FACT_FIELDS.length;

/** The codes of the enumerated facts; 0 is always "none". */
export const DEATH_CAUSES = [undefined, "self", "body", "head-on", "starved"] as const;
export const OUTCOMES = [undefined, "won", "lost", "drawn"] as const;
export const REASONS = [undefined, "last-standing", "time-up", "learner-died"] as const;

/** A whole number, or an inclusive range to draw one from each time a match starts. */
export type Draw = number | [number, number];

/** How one slot of the batch plays, match after match. */
export interface MatchSpec {
  /** Roster ids or brain files, one a seat besides the learner's: 0 (alone) to 3. */
  opponents: string[];
  mode: GameMode | "mixed";
  /** The learner's reaction delay in ticks, and each opponent's. */
  learnerDelay: Draw;
  opponentDelay: Draw;
  /** The learner's seat: a number, or "random" to draw one each match. */
  seat: number | "random";
}

export interface EnvConfig {
  /** Everything the env draws (each match's seed, mode, delays, seat) comes from this. */
  seed: number;
  encoder: EncoderVersion;
  fps: number;
  /** One spec a match: the batch's size is its length. */
  matches: MatchSpec[];
  /** End a match's episode when the learner dies (default true), rather than playing it out among the others. */
  endOnDeath: boolean;
}

/** What one message answers with: the facts and observations of every match, in match order. */
export interface Batch {
  facts: Int32Array;
  observations: Float32Array;
}

const draw = (rng: () => number, value: Draw) =>
  typeof value === "number" ? value : value[0] + Math.floor(rng() * (value[1] - value[0] + 1));

const isWhole = (value: unknown, min: number): value is number => Number.isInteger(value) && (value as number) >= min;

const checkDraw = (name: string, value: unknown) => {
  const ok = Array.isArray(value) ? value.length === 2 && isWhole(value[0], 0) && isWhole(value[1], value[0]) : isWhole(value, 0);
  if (!ok) throw new Error(`${name} must be a whole number of ticks or [min, max], not ${JSON.stringify(value)}`);
};

/** Throws, saying why, unless `raw` is an `EnvConfig`; fills in what has a default. */
export function parseConfig(raw: any): EnvConfig {
  if (typeof raw !== "object" || raw === null) throw new Error("reset needs an object");
  const { seed, encoder = 2, fps = 8, matches, endOnDeath = true } = raw;
  if (!Number.isInteger(seed) || seed < 0 || seed >= 2 ** 32) throw new Error(`seed must be a whole number in [0, 2^32), not ${seed}`);
  if (encoder !== 1 && encoder !== 2) throw new Error(`encoder must be 1 or 2, not ${encoder}`);
  if (typeof fps !== "number" || !(fps > 0)) throw new Error(`fps must be above 0, not ${fps}`);
  if (!Array.isArray(matches) || matches.length < 1) throw new Error("matches must be a list of at least one match spec");
  const specs: MatchSpec[] = matches.map((m: any, i: number) => {
    const at = `matches[${i}]`;
    const { opponents = [], mode = "timed", learnerDelay = 2, opponentDelay = 2, seat = 0 } = m ?? {};
    if (!Array.isArray(opponents) || opponents.length > 3 || opponents.some((o) => typeof o !== "string")) {
      throw new Error(`${at}.opponents must be 0 to 3 roster ids or brain files`);
    }
    if (mode !== "timed" && mode !== "endless" && mode !== "mixed") throw new Error(`${at}.mode must be timed, endless or mixed`);
    checkDraw(`${at}.learnerDelay`, learnerDelay);
    checkDraw(`${at}.opponentDelay`, opponentDelay);
    if (seat !== "random" && !(Number.isInteger(seat) && seat >= 0 && seat <= opponents.length)) {
      throw new Error(`${at}.seat must be "random" or a seat from 0 to ${opponents.length}`);
    }
    return { opponents, mode, learnerDelay, opponentDelay, seat };
  });
  return { seed, encoder, fps, matches: specs, endOnDeath: Boolean(endOnDeath) };
}

interface Slot {
  spec: MatchSpec;
  match: SteppedMatch;
  seat: number;
}

/**
 * Many matches at once, for a learner in another process. `reset` starts every
 * match; `step` plays one tick of each from a batch of actions and, where a
 * match ended, starts the next on a seed drawn from the batch's. All of it is a
 * function of the config and the actions.
 */
export class Env {
  private readonly rng: () => number;
  private readonly slots: Slot[];
  /** Loaded once: a brain file is read a single time however many matches use it. */
  private readonly players = new Map<string, Seat["player"]>();
  readonly obsSize: number;

  constructor(readonly config: EnvConfig) {
    this.rng = mulberry32(config.seed);
    this.obsSize = OBSERVATION_SIZE[config.encoder];
    for (const spec of config.matches) for (const id of spec.opponents) this.player(id);
    this.slots = config.matches.map((spec) => this.start(spec));
  }

  get size() {
    return this.slots.length;
  }

  /** The first observation of every match, and facts saying nothing has happened yet. */
  reset(): Batch {
    const facts = new Int32Array(this.size * FACT_COUNT);
    const observations = new Float32Array(this.size * this.obsSize);
    this.slots.forEach((slot, i) => {
      this.fillNew(slot, facts, observations, i);
      facts[i * FACT_COUNT + FACT_FIELDS.indexOf("alive")] = 1;
    });
    return { facts, observations };
  }

  /** One tick of every match. A match that ended reports its last tick's facts, with `ended` set, and its observation is the next match's first. */
  step(actions: ArrayLike<number>): Batch {
    if (actions.length !== this.size) throw new Error(`step needs ${this.size} actions, got ${actions.length}`);
    for (let i = 0; i < actions.length; i++) {
      if (actions[i] !== 0 && actions[i] !== 1 && actions[i] !== 2) throw new Error(`action ${i} must be 0, 1 or 2, not ${actions[i]}`);
    }
    const facts = new Int32Array(this.size * FACT_COUNT);
    const observations = new Float32Array(this.size * this.obsSize);
    this.slots.forEach((slot, i) => {
      const action = actions[i] as Action;
      const f = slot.match.step(action);
      let reason: number = f.reason ? REASONS.indexOf(f.reason) : 0;
      let outcome: number = f.outcome ? OUTCOMES.indexOf(f.outcome) : 0;
      let ended = f.over;
      if (!ended && !f.alive && this.config.endOnDeath) {
        ended = true;
        reason = REASONS.indexOf("learner-died");
        outcome = OUTCOMES.indexOf("lost");
      }
      this.writeFacts(facts, i, f, slot.seat, ended, outcome, reason);
      if (ended) {
        const next = this.start(slot.spec);
        slot.match = next.match;
        slot.seat = next.seat;
      }
      this.fillObservation(slot, observations, i);
    });
    return { facts, observations };
  }

  private player(id: string) {
    let player = this.players.get(id);
    if (!player) this.players.set(id, (player = loadCandidate(id).player));
    return player;
  }

  private start(spec: MatchSpec): Slot {
    const { rng } = this;
    const seed = Math.floor(rng() * 2 ** 32);
    const mode: GameMode = spec.mode === "mixed" ? (rng() < 0.5 ? "timed" : "endless") : spec.mode;
    const total = spec.opponents.length + 1;
    const seat = spec.seat === "random" ? Math.floor(rng() * total) : spec.seat;
    const learnerDelay = draw(rng, spec.learnerDelay);
    let next = 0;
    const seats: Seat[] = Array.from({ length: total }, (_, i) =>
      i === seat
        ? { player: { name: "learner", decider: () => "r" }, delay: learnerDelay }
        : { player: this.player(spec.opponents[next++]), delay: draw(rng, spec.opponentDelay) });
    const match = new SteppedMatch({ seed, seats, mode, fps: this.config.fps, learner: seat, encoder: this.config.encoder, keepEvents: false });
    return { spec, match, seat };
  }

  private fillNew(slot: Slot, facts: Int32Array, observations: Float32Array, i: number) {
    facts[i * FACT_COUNT + FACT_FIELDS.indexOf("seat")] = slot.seat;
    this.fillObservation(slot, observations, i);
  }

  private fillObservation(slot: Slot, observations: Float32Array, i: number) {
    observations.set(slot.match.observe(), i * this.obsSize);
  }

  private writeFacts(facts: Int32Array, i: number, f: StepFacts, seat: number, ended: boolean, outcome: number, reason: number) {
    const by = f.death?.by;
    const values: Record<(typeof FACT_FIELDS)[number], number> = {
      scoreGained: f.scoreGained,
      ate: f.ate,
      kills: f.kills,
      alive: f.alive ? 1 : 0,
      deathCause: f.death ? DEATH_CAUSES.indexOf(f.death.cause) : 0,
      deathBy: by === undefined ? -1 : Number(by.slice("seat".length)),
      ended: ended ? 1 : 0,
      outcome,
      reason,
      ticks: f.tick,
      seat
    };
    FACT_FIELDS.forEach((name, k) => { facts[i * FACT_COUNT + k] = values[name]; });
  }
}

/** One message: a JSON header and a binary payload, framed as in docs/env-protocol.md. */
export interface Frame {
  header: Record<string, any>;
  payload: Buffer;
}

export const encodeFrame = ({ header, payload }: Frame): Buffer => {
  const json = Buffer.from(JSON.stringify(header), "utf8");
  const prefix = Buffer.alloc(8);
  prefix.writeUInt32LE(json.length, 0);
  prefix.writeUInt32LE(payload.length, 4);
  return Buffer.concat([prefix, json, payload]);
};

/** Cuts a byte stream, arriving in any chunks, into whole frames. */
export class FrameReader {
  private buffer: Buffer = Buffer.alloc(0);

  push(chunk: Buffer): Frame[] {
    this.buffer = this.buffer.length === 0 ? chunk : Buffer.concat([this.buffer, chunk]);
    const frames: Frame[] = [];
    for (;;) {
      if (this.buffer.length < 8) break;
      const jsonLength = this.buffer.readUInt32LE(0);
      const payloadLength = this.buffer.readUInt32LE(4);
      const total = 8 + jsonLength + payloadLength;
      if (this.buffer.length < total) break;
      const header = JSON.parse(this.buffer.subarray(8, 8 + jsonLength).toString("utf8"));
      frames.push({ header, payload: Buffer.from(this.buffer.subarray(8 + jsonLength, total)) });
      this.buffer = this.buffer.subarray(total);
    }
    return frames;
  }
}

/** The state behind one `env` process: answers a request frame with a response frame. */
export class EnvSession {
  private env?: Env;
  closed = false;

  handle({ header, payload }: Frame): Frame {
    try {
      switch (header.type) {
        case "reset": {
          this.env = new Env(parseConfig(header));
          return this.batch(this.env, this.env.reset());
        }
        case "step": {
          if (!this.env) throw new Error("step before reset");
          return this.batch(this.env, this.env.step(payload));
        }
        case "close":
          this.closed = true;
          return { header: { type: "closed" }, payload: Buffer.alloc(0) };
        default:
          throw new Error(`unknown message type ${JSON.stringify(header.type)}`);
      }
    } catch (error) {
      return { header: { type: "error", message: (error as Error).message }, payload: Buffer.alloc(0) };
    }
  }

  private batch(env: Env, { facts, observations }: Batch): Frame {
    const header = {
      type: "batch",
      protocol: PROTOCOL_VERSION,
      matches: env.size,
      obsSize: env.obsSize,
      factFields: FACT_FIELDS
    };
    const payload = Buffer.concat([
      Buffer.from(facts.buffer, facts.byteOffset, facts.byteLength),
      Buffer.from(observations.buffer, observations.byteOffset, observations.byteLength)
    ]);
    return { header, payload };
  }
}
