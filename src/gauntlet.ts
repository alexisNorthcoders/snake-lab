import { readFileSync } from "node:fs";
import { basename } from "node:path";
import { GameMode } from "snake-colyseus/engine";
import { Brain, RosterEntry, brainDecider, brainProblems, roster } from "snake-colyseus/bots";
import { MatchOptions, Player, playMatch } from "./match.ts";

/** The promotion bar: the share of its matches against the rookie at `BAR_DELAY`, both modes together, a candidate must win. */
export const BAR = 0.6;
export const BAR_DELAY = 2;
export const BAR_OPPONENT = "rookie";

/** The whole grid, unless a test asks for less. */
export const DELAYS = [0, 1, 2, 3, 4];
export const MODES: readonly GameMode[] = ["timed", "endless"];

/** The snake on trial: a roster entry (with its id) or a brain file (without one). */
export interface Candidate {
  id?: string;
  player: Player;
}

export interface GauntletOptions {
  candidate: Player;
  /** Played in order: the rookie first (see `opponentsFor`). */
  opponents: RosterEntry[];
  /** The first seed; the rest follow it, wrapping at 2^32. */
  baseSeed: number;
  /** N: seeds per cell. Each is played twice, the candidate in each seat. */
  seeds: number;
  fps: number;
  delays?: number[];
  modes?: readonly GameMode[];
  /** Called before each match is played: for progress, or to see what's played. */
  onMatch?: (options: MatchOptions) => void;
}

/** One opponent, delay and mode: how the candidate's matches there ended. */
export interface GauntletCell {
  /** The opponent's roster id. */
  opponent: string;
  delay: number;
  mode: GameMode;
  matches: number;
  wins: number;
  losses: number;
  draws: number;
}

export interface GauntletReport {
  candidate: string;
  baseSeed: number;
  seeds: number;
  fps: number;
  /** By opponent, then delay, then mode. */
  cells: GauntletCell[];
}

export interface Verdict {
  bar: number;
  /** The candidate's wins ÷ all its matches against the rookie at the bar's delay. */
  winRate: number;
  matches: number;
  pass: boolean;
}

/** A roster id, or the path to a brain file. Throws, saying why, on anything else, listing a brain file's problems. */
export function loadCandidate(name: string, from: RosterEntry[] = roster): Candidate {
  const entry = from.find((e) => e.id === name);
  if (entry) return { id: entry.id, player: entry };

  let text: string;
  try {
    text = readFileSync(name, "utf8");
  } catch {
    throw new Error(`no roster snake or brain file "${name}": the roster has ${from.map((e) => e.id).join(", ")}`);
  }
  let brain: unknown;
  try {
    brain = JSON.parse(text);
  } catch (error) {
    throw new Error(`${name} isn't JSON: ${(error as Error).message}`);
  }
  const problems = brainProblems(brain);
  if (problems.length > 0) throw new Error(`${name} isn't a valid brain:\n${problems.map((p) => `  - ${p}`).join("\n")}`);
  return { player: { name: basename(name), decider: brainDecider(brain as Brain) } };
}

/** Who the candidate plays: the rookie, always first (even against itself), then every other roster entry. */
export const opponentsFor = (candidateId: string | undefined, from: RosterEntry[] = roster): RosterEntry[] =>
  from.filter((e) => e.id === BAR_OPPONENT || e.id !== candidateId)
    .sort((a, b) => Number(b.id === BAR_OPPONENT) - Number(a.id === BAR_OPPONENT));

/**
 * Plays the candidate in 1v1 matches against each opponent, at each delay (both
 * snakes at it) and in each mode: every seed from `baseSeed` on, twice, with
 * the candidate in seat 0 and then seat 1. A match is a win when the candidate
 * is the winner, a loss when the opponent is, and a draw when no one is.
 */
export function runGauntlet(options: GauntletOptions): GauntletReport {
  const { candidate, opponents, baseSeed, seeds, fps, delays = DELAYS, modes = MODES, onMatch } = options;
  const cells: GauntletCell[] = [];
  for (const opponent of opponents) {
    for (const delay of delays) {
      for (const mode of modes) {
        const cell: GauntletCell = { opponent: opponent.id, delay, mode, matches: 0, wins: 0, losses: 0, draws: 0 };
        for (let i = 0; i < seeds; i++) {
          const seed = (baseSeed + i) % 2 ** 32;
          for (const candidateSeat of [0, 1]) {
            const players = candidateSeat === 0 ? [candidate, opponent] : [opponent, candidate];
            const match: MatchOptions = { seed, seats: players.map((player) => ({ player, delay })), mode, fps, keepEvents: false };
            onMatch?.(match);
            const { winnerId } = playMatch(match).result;
            cell.matches++;
            if (winnerId === undefined) cell.draws++;
            else if (winnerId === `seat${candidateSeat}`) cell.wins++;
            else cell.losses++;
          }
        }
        cells.push(cell);
      }
    }
  }
  return { candidate: candidate.name, baseSeed, seeds, fps, cells };
}

/** A cell's win, loss and draw rates, each out of all its matches. */
export const rates = ({ matches, wins, losses, draws }: GauntletCell) =>
  ({ win: wins / matches, loss: losses / matches, draw: draws / matches });

/** Pass or fail against the bar. With no matches against the rookie at the bar's delay, it fails. */
export function verdict(report: GauntletReport): Verdict {
  const counted = report.cells.filter((c) => c.opponent === BAR_OPPONENT && c.delay === BAR_DELAY);
  const matches = counted.reduce((sum, c) => sum + c.matches, 0);
  const wins = counted.reduce((sum, c) => sum + c.wins, 0);
  const winRate = matches === 0 ? 0 : wins / matches;
  return { bar: BAR, winRate, matches, pass: matches > 0 && winRate >= BAR };
}
