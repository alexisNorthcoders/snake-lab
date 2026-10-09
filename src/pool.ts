import { Worker } from "node:worker_threads";
import type { Brain } from "snake-colyseus/bots";
import type { Fixture, TrainSettings } from "./train.ts";

/** What a worker is sent: one snake to play through the generation's fixtures. */
export interface Job {
  index: number;
  brain: Brain;
  fixtures: Fixture[];
  /** The whole generation, which the fixtures' population opponents are picked from. */
  population: Brain[];
  settings: TrainSettings;
}

/** What a worker sends back: the snake's fitness, or why it couldn't play. */
export type Done = { index: number; fitness: number } | { index: number; error: string };

/**
 * A worker thread doesn't get tsx's loader, and Node (22.18 on) strips types
 * natively instead, which can't load a module that imports a type as a value.
 * So each worker boots from a module that registers tsx, then loads worker.ts.
 */
function bootUrl(): URL {
  const boot = `import { register } from ${JSON.stringify(import.meta.resolve("tsx/esm/api"))}; register(); await import(${JSON.stringify(new URL("./worker.ts", import.meta.url).href)});`;
  return new URL(`data:text/javascript,${encodeURIComponent(boot)}`);
}

/**
 * Worker threads that play a generation's snakes, each through every fixture.
 * A worker takes the next snake as soon as it's free, and each fitness goes
 * back in the snake's place, so the number of workers never changes the
 * result. A snake that throws, or a worker that dies, fails the generation.
 */
export class Pool {
  private workers: Worker[];
  private failure: Error | undefined;
  private generation: { onDone: (worker: Worker, done: Done) => void; fail: (error: Error) => void } | undefined;

  constructor(size: number) {
    if (!Number.isInteger(size) || size < 1) throw new Error(`workers must be a whole number of at least 1, not ${size}`);
    this.workers = Array.from({ length: size }, (_, i) => {
      const worker = new Worker(bootUrl());
      worker.on("message", (done: Done) => this.generation?.onDone(worker, done));
      const fail = (error: Error) => {
        this.failure ??= error;
        this.generation?.fail(error);
      };
      worker.on("error", (error) => fail(new Error(`worker ${i} failed: ${error.stack ?? error.message}`)));
      worker.on("exit", (code) => fail(new Error(`worker ${i} stopped (exit code ${code})`)));
      return worker;
    });
  }

  /** Every snake's fitness over the fixtures, in the population's order. */
  fitness(population: Brain[], fixtures: Fixture[], settings: TrainSettings): Promise<number[]> {
    if (this.failure) return Promise.reject(this.failure);
    if (this.generation) return Promise.reject(new Error("the pool is already playing a generation"));
    return new Promise((resolve, reject) => {
      const fitness: number[] = new Array(population.length);
      let next = 0;
      let left = population.length;
      const send = (worker: Worker) => {
        if (next === population.length) return;
        const index = next++;
        const job: Job = { index, brain: population[index], fixtures, population, settings };
        worker.postMessage(job);
      };
      this.generation = {
        onDone: (worker, done) => {
          if ("error" in done) {
            this.generation?.fail(new Error(`snake ${done.index} couldn't be played: ${done.error}`));
            return;
          }
          fitness[done.index] = done.fitness;
          if (--left === 0) {
            this.generation = undefined;
            resolve(fitness);
          } else send(worker);
        },
        fail: (error) => {
          this.failure ??= error;
          this.generation = undefined;
          next = population.length; // no new jobs
          void this.close();
          reject(error);
        }
      };
      this.workers.forEach(send);
    });
  }

  async close() {
    const workers = this.workers;
    if (workers.length === 0) return;
    this.workers = [];
    this.failure ??= new Error("the pool is closed");
    await Promise.all(workers.map((worker) => {
      worker.removeAllListeners("exit");
      return worker.terminate();
    }));
  }
}
