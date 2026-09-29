import { GameMode } from "snake-colyseus/engine";
import { roster } from "snake-colyseus/bots";
import { loadCandidate } from "../gauntlet.ts";
import { MatchOptions, Seat } from "../match.ts";

const rosterIds = roster.map((entry) => entry.id).join(", ");

/** The options `match` and `bench` share, as `parseArgs` takes them. */
export const matchArgs = {
  seed: { type: "string", default: "1" },
  players: { type: "string", default: "rookie,rookie" },
  mode: { type: "string", default: "timed" },
  fps: { type: "string", default: "8" },
  delays: { type: "string", default: "2" },
  help: { type: "boolean" }
} as const;

export const matchArgsHelp = `  --seed <n>        the seed (bench: the first), a whole number in [0, 2^32) (default 1)
  --players <ids>   roster ids or brain files in seat order, comma-separated (default rookie,rookie)
                    roster: ${rosterIds}
  --mode <mode>     timed or endless (default timed)
  --fps <n>         ticks per second, which sets a timed round's length (default 8)
  --delays <n,...>  each player's reaction delay in ticks, or one for all (default 2)`;

/** Runs `body`, exiting with its error and `usage` if it throws. */
export function orExit<T>(usage: string, body: () => T): T {
  try {
    return body();
  } catch (error) {
    exit(`${(error as Error).message}\n\n${usage}`);
  }
}

export function exit(message: string, code = 1): never {
  (code === 0 ? console.log : console.error)(message);
  process.exit(code);
}

export const parseNumber = (name: string, value: string) => {
  const n = Number(value);
  if (value.trim() === "" || !Number.isFinite(n)) throw new Error(`--${name} must be a number, not "${value}"`);
  return n;
};

const parseMode = (value: string): GameMode => {
  if (value !== "timed" && value !== "endless") throw new Error(`--mode must be timed or endless, not "${value}"`);
  return value;
};

/** The seats named by `--players` (roster ids or brain files) and `--delays`. The match itself checks how many there are. */
const parseSeats = (players: string, delays: string): Seat[] => {
  const ids = players.split(",").map((id) => id.trim());
  const perSeat = delays.split(",").map((d) => parseNumber("delays", d));
  if (perSeat.length !== 1 && perSeat.length !== ids.length) {
    throw new Error(`--delays gives ${perSeat.length} delays for ${ids.length} players: give one, or one each`);
  }
  return ids.map((id, i) => ({ player: loadCandidate(id).player, delay: perSeat.length === 1 ? perSeat[0] : perSeat[i] }));
};

/** The match the shared options describe. Throws, saying why, on one that can't be read. */
export const matchOptions = (args: { seed: string; players: string; delays: string; mode: string; fps: string }): MatchOptions => ({
  seed: parseNumber("seed", args.seed),
  seats: parseSeats(args.players, args.delays),
  mode: parseMode(args.mode),
  fps: parseNumber("fps", args.fps)
});

/** The options `gauntlet` and `promote` share, as `parseArgs` takes them. */
export const gauntletArgs = {
  seeds: { type: "string", default: "100" },
  "base-seed": { type: "string", default: "1" },
  fps: { type: "string", default: "8" }
} as const;

export const gauntletArgsHelp = `  --seeds <n>       seeds per opponent, delay and mode, each played in both seats (default 100)
  --base-seed <n>   the first seed, a whole number in [0, 2^32) (default 1)
  --fps <n>         ticks per second, which sets a timed round's length (default 8)`;

/** The gauntlet's settings from the shared options. Throws, saying why, on one that can't be used. */
export function gauntletSettings(values: { seeds: string; "base-seed": string; fps: string }) {
  const seeds = parseNumber("seeds", values.seeds);
  if (!Number.isInteger(seeds) || seeds < 1) throw new Error(`--seeds must be a whole number above 0, not ${seeds}`);
  const baseSeed = parseNumber("base-seed", values["base-seed"]);
  if (!Number.isInteger(baseSeed) || baseSeed < 0 || baseSeed >= 2 ** 32) {
    throw new Error(`--base-seed must be a whole number in [0, 2^32), not ${baseSeed}`);
  }
  const fps = parseNumber("fps", values.fps);
  if (fps <= 0) throw new Error(`--fps must be above 0, not ${fps}`);
  return { seeds, baseSeed, fps };
}
