import { parseArgs } from "node:util";
import { DiedEvent } from "snake-colyseus/engine";
import { MatchEvent, MatchPlayer, playMatch } from "../match.ts";
import { exit, matchArgs, matchArgsHelp, matchOptions, orExit } from "./args.ts";

const usage = `Plays one match and prints the result.

Usage: npm run match -- [options]

${matchArgsHelp}
  --events          also print every event, tick by tick
  --json            print the whole result as JSON instead`;

const args = orExit(usage, () => parseArgs({
  options: { ...matchArgs, events: { type: "boolean" }, json: { type: "boolean" } },
  strict: true
}).values);
if (args.help) exit(usage, 0);

const options = orExit(usage, () => matchOptions(args));
const result = orExit(usage, () => playMatch(options));

if (args.json) {
  console.log(JSON.stringify(result, null, 2));
  process.exit(0);
}

const byId = new Map(result.players.map((player) => [player.id, player]));
const label = (id: string) => `${byId.get(id)!.name} (${id})`;
const how = (death: DiedEvent) => `${death.cause}${death.by ? ` into ${label(death.by)}` : ""}`;
const describe = (event: MatchEvent) => event.kind === "ate"
  ? `${label(event.player)} ate a ${event.food.type} at ${event.food.x},${event.food.y} for ${event.score}`
  : `${label(event.player)} died: ${how(event)}`;
const deathOf = (player: MatchPlayer) => {
  const death = result.events!.find((event) => event.kind === "died" && event.player === player.id);
  return death?.kind === "died" ? `died on tick ${death.tick}: ${how(death)}` : "alive";
};

const { reason, winnerId } = result.result;
console.log(`seed ${options.seed}, ${options.mode} at ${options.fps} fps: ${result.ticks} ticks, ended ${reason}, ` +
  (winnerId ? `won by ${label(winnerId)}` : "no winner"));
result.players.forEach((player, i) => {
  const errors = player.decisionErrors > 0 ? `, ${player.decisionErrors} decisions threw` : "";
  console.log(`  ${player.id}  ${player.name.padEnd(12)} delay ${options.seats[i].delay}  score ${String(player.score).padStart(4)}  ` +
    `length ${String(player.length).padStart(3)}  ${deathOf(player)}${errors}`);
});
if (args.events) result.events!.forEach((event) => console.log(`  tick ${String(event.tick).padStart(4)}  ${describe(event)}`));
