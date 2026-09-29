import { existsSync, readFileSync } from "node:fs";
import { Brain, brainDecider } from "snake-colyseus/bots";
import { GameMode } from "snake-colyseus/engine";
import { Frame, MatchResult, playMatch } from "../match.ts";
import { runFiles } from "../run.ts";
import { Fixture, TrainSettings, drawGeneration, fixtureOptions } from "../train.ts";

/** The fixtures the grid can show: those against a population opponent are left out, as that population isn't on disk. */
export const replayable = (fixtures: Fixture[]): Fixture[] => fixtures.filter(({ opponents }) => opponents.every((o) => "roster" in o));

/** What a game is labelled with. */
export interface GameInfo {
  /** The fixture's place among the generation's, so a label can be traced back to it. */
  index: number;
  seed: number;
  mode: GameMode;
  seat: number;
  /** One a seat, in seat order. */
  delays: number[];
  /** Each seat's name, in seat order. */
  names: string[];
}

/** One game as the page gets it: a start, every tick's board (0 is the board as dealt), then the end. */
export type GameMessage =
  | { kind: "start"; game: GameInfo }
  | { kind: "tick"; frame: Frame }
  | { kind: "end"; result: MatchResult["result"]; ticks: number; players: { id: string; name: string; score: number; length: number }[] };

/** Plays `brain` in `fixture` (which mustn't have a population opponent) and yields its tick stream. */
export function* gameStream(fixture: Fixture, index: number, brain: Brain, fps: number): Generator<GameMessage> {
  const frames: Frame[] = [];
  const options = fixtureOptions(fixture, { name: "Candidate", decider: brainDecider(brain) }, [], fps);
  const match = playMatch({ ...options, onTick: (frame) => frames.push(frame) });
  const { seed, mode, seat, delays } = fixture;
  yield { kind: "start", game: { index, seed, mode, seat, delays, names: options.seats.map(({ player }) => player.name) } };
  for (const frame of frames) yield { kind: "tick", frame };
  yield { kind: "end", result: match.result, ticks: match.ticks, players: match.players.map(({ id, name, score, length }) => ({ id, name, score, length })) };
}

/** The generation's replayable fixtures, the same the trainer played: at most `limit` of them, with their places among all. */
export function sampleFixtures(settings: TrainSettings, g: number, limit = Infinity): { fixture: Fixture; index: number }[] {
  return drawGeneration(settings, g).fixtures
    .map((fixture, index) => ({ fixture, index }))
    .filter(({ fixture }) => replayable([fixture]).length > 0)
    .slice(0, limit);
}

/** Generation `g`'s fittest, from the run folder; undefined if it isn't there (yet). */
export function readGenerationBest(run: string, g: number): Brain | undefined {
  const path = runFiles(run).generation(g);
  if (!existsSync(path)) return undefined;
  try {
    return JSON.parse(readFileSync(path, "utf8"));
  } catch {
    return undefined;
  }
}
