// Prints the pinned package's roster ids as a JSON array, for the Python trainer's league.
import { roster } from "snake-colyseus/bots";

console.log(JSON.stringify(roster.map((entry) => entry.id).sort()));
