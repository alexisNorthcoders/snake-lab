// Reads {brain, inputs} as JSON on stdin; prints what the package says: the brain's problems, its outputs for each
// input (`forward`), the move `brainDecider` semantics pick (argmax, a tie to straight, then left), and the engine's rules version.
import { readFileSync } from "node:fs";
import { brainProblems, forward } from "snake-colyseus/bots";
import { RULES_VERSION } from "snake-colyseus/engine";

const { brain, inputs } = JSON.parse(readFileSync(0, "utf8"));
const problems = brainProblems(brain);
const outputs = problems.length ? [] : inputs.map((input: number[]) => forward(brain, input));
console.log(JSON.stringify({ problems, outputs, rulesVersion: RULES_VERSION }));
