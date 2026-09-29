import { parentPort } from "node:worker_threads";
import type { Done, Job } from "./pool.ts";
import { evaluate } from "./train.ts";

/** A `Pool`'s worker: plays each snake it's sent through the fixtures and sends back its fitness. */
parentPort!.on("message", ({ index, brain, fixtures, population, settings }: Job) => {
  let done: Done;
  try {
    done = { index, fitness: evaluate(brain, fixtures, population, settings) };
  } catch (error) {
    done = { index, error: (error as Error).stack ?? String(error) };
  }
  parentPort!.postMessage(done);
});
