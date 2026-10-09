import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { type FoodPlacement, layFood, mulberry32, roundTicks } from "snake-colyseus/engine";
import { type BotView, type Decider, pickBot } from "snake-colyseus/bots";
import { type MatchOptions, type Seat, playMatch } from "../src/match.ts";

const rookie = pickBot("rookie");
const dummy = pickBot("dummy");

const straight: Decider = (view) => {
  const { x, y } = view.self.movedDirection;
  return x === 1 ? "r" : x === -1 ? "l" : y === 1 ? "d" : y === -1 ? "u" : "r";
};

/** A decider that plays like `play` and keeps every view it was given. */
const spy = (play: Decider = straight) => {
  const views: BotView[] = [];
  const decider: Decider = (view) => {
    views.push(view);
    return play(view);
  };
  return { views, decider };
};

const seats = (...players: Seat["player"][]): Seat[] => players.map((player) => ({ player, delay: 2 }));

const options = (overrides: Partial<MatchOptions> = {}): MatchOptions => ({
  seed: 1,
  seats: seats(rookie, dummy),
  mode: "timed",
  fps: 8,
  ...overrides
});

describe("playMatch", () => {
  it("plays the same match from the same seed and players", () => {
    for (const mode of ["timed", "endless"] as const) {
      const first = playMatch(options({ seed: 42, mode }));
      const again = playMatch(options({ seed: 42, mode }));
      assert.deepEqual(again, first);
      assert.ok(first.events!.length > 0);
    }
  });

  it("plays a different match from a different seed", () => {
    const first = playMatch(options({ seed: 1 }));
    const other = playMatch(options({ seed: 2 }));
    assert.notDeepEqual(other, first);
  });

  it("lays the food from the seed first, as the room does", () => {
    const watcher = spy();
    playMatch(options({ seed: 99, seats: [{ player: { name: "Watcher", decider: watcher.decider }, delay: 0 }] }));
    const room = { foodCoordinates: [] as FoodPlacement[] };
    layFood(room, mulberry32(99), (placement) => placement);
    assert.deepEqual(watcher.views[0].food.map(({ x, y, type }) => ({ x, y, type })), room.foodCoordinates.map(({ x, y, type }) => ({ x, y, type })));
  });

  it("plays rookie against Dummy to the end in timed and in endless", () => {
    for (const mode of ["timed", "endless"] as const) {
      const { result, ticks, players } = playMatch(options({ mode }));
      assert.ok(["last-standing", "time-up"].includes(result.reason), `${mode}: ${result.reason}`);
      assert.ok(ticks > 0);
      assert.deepEqual(players.map((p) => p.name), ["Rookie", "Dummy"]);
    }
  });

  it("reports each player's final score and length, and names a winner among them", () => {
    const { result, players, events } = playMatch(options({ seed: 7 }));
    players.forEach((player) => {
      const eaten = events!.filter((e) => e.kind === "ate" && e.player === player.id);
      assert.equal(player.score, eaten.reduce((sum, e) => sum + (e.kind === "ate" ? e.score : 0), 0));
      assert.equal(player.length, 1 + eaten.length);
    });
    if (result.winnerId !== undefined) assert.ok(players.some((p) => p.id === result.winnerId));
  });

  it("records each event with the tick it happened on", () => {
    const { ticks, events } = playMatch(options({ seed: 3 }));
    const tickNumbers = events!.map((e) => e.tick);
    assert.deepEqual(tickNumbers, [...tickNumbers].sort((a, b) => a - b));
    assert.ok(tickNumbers.every((t) => t >= 1 && t <= ticks));
  });

  it("skips keeping events when asked, and plays the same match", () => {
    const kept = playMatch(options({ seed: 5 }));
    const skipped = playMatch(options({ seed: 5, keepEvents: false }));
    assert.equal(skipped.events, undefined);
    const { events, ...rest } = kept;
    assert.deepEqual(skipped, rest);
  });

  it("limits a timed match at fps 8 to 1440 ticks, the engine's limit", () => {
    const watcher = spy(rookie.kind === "scripted" ? rookie.decider : straight);
    const { ticks } = playMatch(options({ seats: [{ player: { name: "Watcher", decider: watcher.decider }, delay: 2 }, { player: dummy, delay: 2 }] }));
    assert.equal(roundTicks(8), 1440);
    assert.equal(watcher.views[0].tickLimit, 1440);
    assert.equal(watcher.views[0].ticksLeft, 1440);
    assert.ok(ticks <= 1440);
  });

  it("changes a timed match's limit with the fps, and ends it on time", () => {
    for (const fps of [0.5, 1]) {
      // A lone snake going straight can't eat enough in this time to crash.
      const lone = spy();
      const { result, ticks } = playMatch(options({ fps, seats: [{ player: { name: "Lone", decider: lone.decider }, delay: 0 }] }));
      assert.equal(lone.views[0].tickLimit, roundTicks(fps));
      assert.equal(result.reason, "time-up");
      assert.equal(ticks, roundTicks(fps));
    }
  });

  it("applies each player's delay separately", () => {
    // Seats 0 and 1 start in the same row, ten cells apart, and go straight
    // at the same speed, so neither ever reaches the other.
    const now = spy();
    const late = spy();
    playMatch(options({
      fps: 1,
      seats: [
        { player: { name: "Now", decider: now.decider }, delay: 0 },
        { player: { name: "Late", decider: late.decider }, delay: 4 }
      ]
    }));

    assert.ok(now.views.length > 10);
    now.views.forEach((view, t) => {
      assert.deepEqual(view.others[0].head, late.views[t].self.head, `delay 0 at tick ${t}`);
    });
    late.views.forEach((view, t) => {
      assert.deepEqual(view.others[0].head, now.views[Math.max(0, t - 4)].self.head, `delay 4 at tick ${t}`);
    });
    assert.notDeepEqual(late.views[10].others[0].head, now.views[10].self.head);
  });

  it("keeps a throwing decider's direction instead of ending the match", () => {
    const watcher = spy();
    const thrower: Decider = () => { throw new Error("boom"); };
    const { result, ticks, players } = playMatch(options({
      seats: [
        { player: { name: "Thrower", decider: thrower }, delay: 0 },
        { player: { name: "Watcher", decider: watcher.decider }, delay: 0 }
      ]
    }));

    assert.ok(["last-standing", "time-up"].includes(result.reason));
    assert.ok(ticks > 1);
    assert.equal(players[0].decisionErrors, ticks);
    assert.equal(players[1].decisionErrors, 0);
    // The thrower keeps heading right, as it started.
    const seen = watcher.views.map((view) => view.others[0].head);
    seen.slice(1).forEach((head, t) => assert.equal(head.x, (seen[t].x + 1) % 20));
  });

  it("seats players in the spawn table's order", () => {
    const watchers = [spy(), spy(), spy(), spy()];
    playMatch(options({
      fps: 1,
      seats: watchers.map((w, i) => ({ player: { name: `W${i}`, decider: w.decider }, delay: 0 }))
    }));
    assert.deepEqual(watchers.map((w) => w.views[0].self.head), [{ x: 5, y: 5 }, { x: 15, y: 5 }, { x: 5, y: 15 }, { x: 15, y: 15 }]);
  });

  it("refuses a match it can't play", () => {
    assert.throws(() => playMatch(options({ seats: [] })), /1 to 4 players/);
    assert.throws(() => playMatch(options({ seats: seats(rookie, rookie, rookie, rookie, rookie) })), /1 to 4 players/);
    assert.throws(() => playMatch(options({ seed: -1 })), /seed/);
    assert.throws(() => playMatch(options({ seed: 1.5 })), /seed/);
    assert.throws(() => playMatch(options({ fps: 0 })), /fps/);
    assert.throws(() => playMatch(options({ seats: [{ player: rookie, delay: -1 }] })), /delay/);
  });
});
