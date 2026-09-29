import {
  Cell,
  DeathCause,
  Direction,
  FoodPlacement,
  GameMode,
  PlayerShape,
  RoundEndReason,
  SnakeShape,
  TickEvent,
  beginPlay,
  dealRound,
  directionMap,
  layFood,
  mulberry32,
  newPlainCell,
  roundTicks,
  startingPositions,
  tailCells,
  tick,
  turn
} from "snake-colyseus/engine";
import { BotView, Decider, ENCODER_SIZE, ENCODER_V2_SIZE, RosterEntry, Snapshots, deciderFor, encode, encodeV2, viewFor } from "snake-colyseus/bots";

/** Who plays a seat: a roster entry, or any decider with a name (a brain that isn't in the roster, say). */
export type Player = RosterEntry | { name: string; decider: Decider };

/** A seat at the board: its player, and how many ticks old its view of the other snakes is. */
export interface Seat {
  player: Player;
  delay: number;
}

export interface MatchOptions {
  /** Every random rule draws from this, through the engine's generator: a whole number in [0, 2^32). */
  seed: number;
  /** 1 to 4 seats. The first starts in the spawn table's first corner, and so on. */
  seats: Seat[];
  mode: GameMode;
  /** Ticks per second: sets a timed round's tick limit, never how fast the match runs. */
  fps: number;
  /** Keep every tick's events (the default), or only the result. */
  keepEvents?: boolean;
  /** Called with the board before the first tick (tick 0) and after every tick. Only looks: never changes a match. */
  onTick?: (frame: Frame) => void;
}

/** The board at the end of a tick: each snake's cells (head first), the food, and what happened on that tick. */
export interface Frame {
  /** 0 is the board as dealt. */
  tick: number;
  snakes: { id: string; cells: Cell[]; score: number; dead: boolean }[];
  food: { x: number; y: number; type: string }[];
  events: MatchEvent[];
}

/** An engine event, with the tick it happened on, counting from 1. */
export type MatchEvent = TickEvent & { tick: number };

export interface MatchPlayer {
  /** `seat0` to `seat3`: what the result and the events name the player by. */
  id: string;
  name: string;
  score: number;
  /** Head included. */
  length: number;
  /** Ticks its decider threw on, keeping its direction. */
  decisionErrors: number;
}

export interface MatchResult {
  /** The engine's round-over report: why the round ended, and its winner if there's one. */
  result: { reason: RoundEndReason; winnerId?: string };
  ticks: number;
  /** In seat order. */
  players: MatchPlayer[];
  /** Left out when `keepEvents` is false. */
  events?: MatchEvent[];
}

interface Bot {
  shape: PlayerShape<Cell>;
  name: string;
  decider: Decider;
  snapshots: Snapshots;
  decisionErrors: number;
}

const newSnake = (): SnakeShape<Cell> => ({
  x: 0,
  y: 0,
  direction: { x: 1, y: 0 },
  movedDirection: { x: 1, y: 0 },
  size: 1,
  score: 0,
  hunger: 0,
  isDead: false,
  tail: [],
  tailCursor: 0
});

const deciderOf = (player: Player): Decider => ("kind" in player ? deciderFor(player) : player.decider);

/** Throws, saying why, if `options` isn't a match that can be played. */
export const checkMatchOptions = (options: MatchOptions) => {
  const { seed, seats, fps } = options;
  if (!Number.isInteger(seed) || seed < 0 || seed >= 2 ** 32) {
    throw new Error(`seed must be a whole number in [0, 2^32), not ${seed}`);
  }
  if (seats.length < 1 || seats.length > startingPositions.length) {
    throw new Error(`a match takes 1 to ${startingPositions.length} players, not ${seats.length}`);
  }
  if (!Number.isFinite(fps) || fps <= 0) throw new Error(`fps must be above 0, not ${fps}`);
  seats.forEach(({ delay }, i) => {
    if (!Number.isInteger(delay) || delay < 0) throw new Error(`seat ${i}'s delay must be a whole number of ticks, not ${delay}`);
  });
};

/** A move for the learner, relative to the way it last moved. */
export const LEFT = 0;
export const STRAIGHT = 1;
export const RIGHT = 2;
export type Action = typeof LEFT | typeof STRAIGHT | typeof RIGHT;

/** Which encoder turns the learner's view into its observation. */
export type EncoderVersion = 1 | 2;

/** How many numbers an observation is, by encoder. */
export const OBSERVATION_SIZE: Record<EncoderVersion, number> = { 1: ENCODER_SIZE, 2: ENCODER_V2_SIZE };

/** What a round says about the learner when it's over. A solo round its snake dies in has no winner: `drawn`. */
export type Outcome = "won" | "lost" | "drawn";

/** A match's options, and which seat is played from outside, if any. */
export interface SteppedOptions extends MatchOptions {
  /** The seat whose moves come from `step`. Left out, every seat plays its own decider. Its `player` is never asked. */
  learner?: number;
  /** How `observe` encodes the learner's view (default 1). */
  encoder?: EncoderVersion;
}

/** What one tick did to the learner: facts, never rewards. */
export interface StepFacts {
  /** The tick just played, counting from 1. */
  tick: number;
  /** The learner's score after the tick minus before it: negative when an endless round drained it. */
  scoreGained: number;
  /** Pellets it ate this tick. */
  ate: number;
  alive: boolean;
  /** Set on the tick it died. */
  death?: { cause: DeathCause; by?: string };
  /** Snakes it killed this tick, by the lab's kill rule: a head-on kills for neither. */
  kills: number;
  /** Whether the round is over. */
  over: boolean;
  /** On the last tick: how the round went for the learner, and why it ended. */
  outcome?: Outcome;
  reason?: RoundEndReason;
}

/**
 * A match you can step. It's `playMatch`'s round, tick by tick and in the same
 * order: every seat records its snapshots, every live seat steers (the learner
 * from the action it's given, the rest from their deciders), the engine runs
 * the tick. Between ticks, `observe` gives the learner's view as it will be
 * when it chooses.
 */
export class SteppedMatch {
  readonly options: SteppedOptions;
  /** Ticks played. */
  ticks = 0;
  /** Set on the last tick. */
  result?: MatchResult;

  private readonly rng: () => number;
  private readonly bots: Bot[];
  private readonly game: {
    players: PlayerShape<Cell>[];
    foodCoordinates: FoodPlacement[];
    aliveCount: number;
    mode: GameMode;
    ticksLeft: number | undefined;
    tickLimit: number | undefined;
  };
  private readonly events: MatchEvent[] = [];

  constructor(options: SteppedOptions) {
    checkMatchOptions(options);
    const { learner, encoder = 1 } = options;
    if (learner !== undefined && (!Number.isInteger(learner) || learner < 0 || learner >= options.seats.length)) {
      throw new Error(`learner must be a seat from 0 to ${options.seats.length - 1}, not ${learner}`);
    }
    if (encoder !== 1 && encoder !== 2) throw new Error(`encoder must be 1 or 2, not ${encoder}`);
    this.options = options;
    const { seed, seats, mode, fps, onTick } = options;

    this.rng = mulberry32(seed);
    this.bots = seats.map(({ player, delay }, i) => ({
      shape: { id: `seat${i}`, snake: newSnake() },
      name: player.name,
      decider: i === learner ? () => { throw new Error("the learner's moves come from step()"); } : deciderOf(player),
      snapshots: new Snapshots(delay),
      decisionErrors: 0
    }));
    this.game = {
      players: this.bots.map((bot) => bot.shape),
      foodCoordinates: [],
      aliveCount: 0,
      mode,
      ticksLeft: undefined,
      tickLimit: undefined
    };

    layFood(this.game, this.rng, (placement) => placement);
    dealRound(this.game, 0, newPlainCell);
    beginPlay(this.game, roundTicks(fps));

    onTick?.(this.frame(0, []));
    this.record();
  }

  get over() {
    return this.result !== undefined;
  }

  /** The learner's view as it will be when it chooses its next move. */
  view(): BotView {
    const bot = this.learnerBot();
    return viewFor(this.game, bot.shape, bot.snapshots);
  }

  /** The learner's observation: its view through the encoder. */
  observe(out?: number[]): number[] {
    const view = this.view();
    return this.options.encoder === 2 ? encodeV2(view, out) : encode(view, out);
  }

  /** Plays one tick. With a learner, `action` is its move. Throws once the round is over. */
  step(action?: Action): StepFacts {
    if (this.result) throw new Error("the round is over");
    const { learner, keepEvents = true, onTick } = this.options;
    const me = learner === undefined ? undefined : this.bots[learner];
    if (me) {
      if (action !== LEFT && action !== STRAIGHT && action !== RIGHT) throw new Error(`action must be 0, 1 or 2, not ${action}`);
      if (!me.shape.snake.isDead) steer(me.shape.snake, directionFor(me.shape.snake.movedDirection, action));
    }
    const scoreBefore = me?.shape.snake.score ?? 0;

    this.bots.forEach((bot) => {
      if (bot === me || bot.shape.snake.isDead) return;
      try {
        steer(bot.shape.snake, bot.decider(viewFor(this.game, bot.shape, bot.snapshots)));
      } catch {
        bot.decisionErrors++;
      }
    });

    const ticks = ++this.ticks;
    const report = tick(this.game, this.rng, newPlainCell);
    if (keepEvents) report.events.forEach((event) => this.events.push({ ...event, tick: ticks }));
    onTick?.(this.frame(ticks, report.events.map((event) => ({ ...event, tick: ticks }))));

    const facts: StepFacts = { tick: ticks, scoreGained: 0, ate: 0, alive: true, kills: 0, over: report.roundOver };
    if (me) {
      const id = me.shape.id;
      facts.scoreGained = me.shape.snake.score - scoreBefore;
      facts.alive = !me.shape.snake.isDead;
      for (const event of report.events) {
        if (event.kind === "ate") {
          if (event.player === id) facts.ate++;
        } else if (event.player === id) {
          facts.death = { cause: event.cause, ...(event.by !== undefined && { by: event.by }) };
        } else if (event.by === id && event.cause !== "head-on") {
          facts.kills++;
        }
      }
    }

    if (report.roundOver) {
      const { reason, winnerId } = report;
      facts.reason = reason!;
      if (me) facts.outcome = winnerId === me.shape.id ? "won" : winnerId === undefined ? "drawn" : "lost";
      this.result = {
        result: { reason: reason!, ...(winnerId !== undefined && { winnerId }) },
        ticks,
        players: this.bots.map(({ shape: { id, snake }, name, decisionErrors }) =>
          ({ id, name, score: snake.score, length: snake.size, decisionErrors })),
        ...(keepEvents && { events: this.events })
      };
    } else {
      this.record();
    }
    return facts;
  }

  private learnerBot(): Bot {
    const { learner } = this.options;
    if (learner === undefined) throw new Error("this match has no learner");
    return this.bots[learner];
  }

  /** Every seat notes where every snake is, before any of them steers. */
  private record() {
    this.bots.forEach((bot) => bot.snapshots.record(this.game));
  }

  private frame(tick: number, events: MatchEvent[]): Frame {
    return {
      tick,
      snakes: this.bots.map(({ shape: { id, snake } }) =>
        ({ id, cells: [{ x: snake.x, y: snake.y }, ...tailCells(snake).map(({ x, y }) => ({ x, y }))], score: snake.score, dead: snake.isDead })),
      food: [...this.game.foodCoordinates].map(({ x, y, type }) => ({ x, y, type })),
      events
    };
  }
}

/**
 * The direction `action` means for a snake that last moved `moved` (right, if
 * it hasn't moved, as the encoders have it).
 */
export function directionFor(moved: { x: number; y: number }, action: Action): Direction {
  const [fx, fy] = moved.x === 0 && moved.y === 0 ? [1, 0] : [moved.x, moved.y];
  // Right of the way it's heading is (-fy, fx); left is its negative.
  const [dx, dy] = action === STRAIGHT ? [fx, fy] : action === RIGHT ? [-fy, fx] : [fy, -fx];
  return dx === 1 ? "r" : dx === -1 ? "l" : dy === 1 ? "d" : "u";
}

/** The action that makes `key` from a snake that last moved `moved`; a reversal, which the engine ignores, is straight on. */
export function actionFor(moved: { x: number; y: number }, key: Direction): Action {
  return ([LEFT, RIGHT] as const).find((action) => directionFor(moved, action) === key) ?? STRAIGHT;
}

/**
 * Plays one round between bots, headless and as fast as it can, in the live
 * room's order: the food is laid from the seed, the snakes dealt their spawns
 * in seat order, and play begins with a timed round's limit at `fps`. Then each
 * tick every bot records where every snake is into its own snapshots, every
 * live bot steers from its view, and the engine runs the tick. As in the room,
 * a decider that throws keeps its direction. It's a `SteppedMatch` with no
 * learner, stepped to the end.
 *
 * Nothing reads the clock or `Math.random`, so the same options always play
 * the same match. An endless round always ends: a snake that stops eating
 * starves, and one that keeps eating fills the board.
 */
export function playMatch(options: MatchOptions): MatchResult {
  const match = new SteppedMatch(options);
  while (!match.over) match.step();
  return match.result!;
}

/** As the room turns a snake: anything that isn't a direction is ignored, and the engine ignores reversals. */
const steer = (snake: SnakeShape<Cell>, key: unknown) => {
  if (typeof key !== "string" || !Object.hasOwn(directionMap, key)) return;
  turn(snake, key as Direction);
};
