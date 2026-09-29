import {
  Cell,
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
import { Decider, RosterEntry, Snapshots, deciderFor, viewFor } from "snake-colyseus/bots";

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

/**
 * Plays one round between bots, headless and as fast as it can, in the live
 * room's order: the food is laid from the seed, the snakes dealt their spawns
 * in seat order, and play begins with a timed round's limit at `fps`. Then each
 * tick every bot records where every snake is into its own snapshots, every
 * live bot steers from its view, and the engine runs the tick. As in the room,
 * a decider that throws keeps its direction.
 *
 * Nothing reads the clock or `Math.random`, so the same options always play
 * the same match. An endless round always ends: a snake that stops eating
 * starves, and one that keeps eating fills the board.
 */
export function playMatch(options: MatchOptions): MatchResult {
  checkMatchOptions(options);
  const { seed, seats, mode, fps, keepEvents = true, onTick } = options;

  const rng = mulberry32(seed);
  const bots: Bot[] = seats.map(({ player, delay }, i) => ({
    shape: { id: `seat${i}`, snake: newSnake() },
    name: player.name,
    decider: deciderOf(player),
    snapshots: new Snapshots(delay),
    decisionErrors: 0
  }));
  const game = {
    players: bots.map((bot) => bot.shape),
    foodCoordinates: [] as FoodPlacement[],
    aliveCount: 0,
    mode,
    ticksLeft: undefined as number | undefined,
    tickLimit: undefined as number | undefined
  };

  layFood(game, rng, (placement) => placement);
  dealRound(game, 0, newPlainCell);
  beginPlay(game, roundTicks(fps));

  const frame = (tick: number, events: MatchEvent[]): Frame => ({
    tick,
    snakes: bots.map(({ shape: { id, snake } }) =>
      ({ id, cells: [{ x: snake.x, y: snake.y }, ...tailCells(snake).map(({ x, y }) => ({ x, y }))], score: snake.score, dead: snake.isDead })),
    food: [...game.foodCoordinates].map(({ x, y, type }) => ({ x, y, type })),
    events
  });
  onTick?.(frame(0, []));

  const events: MatchEvent[] = [];
  for (let ticks = 1; ; ticks++) {
    bots.forEach((bot) => bot.snapshots.record(game));
    bots.forEach((bot) => {
      if (bot.shape.snake.isDead) return;
      try {
        steer(bot.shape.snake, bot.decider(viewFor(game, bot.shape, bot.snapshots)));
      } catch {
        bot.decisionErrors++;
      }
    });

    const report = tick(game, rng, newPlainCell);
    if (keepEvents) report.events.forEach((event) => events.push({ ...event, tick: ticks }));
    onTick?.(frame(ticks, report.events.map((event) => ({ ...event, tick: ticks }))));

    if (report.roundOver) {
      const { reason, winnerId } = report;
      return {
        result: { reason: reason!, ...(winnerId !== undefined && { winnerId }) },
        ticks,
        players: bots.map(({ shape: { id, snake }, name, decisionErrors }) =>
          ({ id, name, score: snake.score, length: snake.size, decisionErrors })),
        ...(keepEvents && { events })
      };
    }
  }
}

/** As the room turns a snake: anything that isn't a direction is ignored, and the engine ignores reversals. */
const steer = (snake: SnakeShape<Cell>, key: unknown) => {
  if (typeof key !== "string" || !Object.hasOwn(directionMap, key)) return;
  turn(snake, key as Direction);
};
