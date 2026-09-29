import { existsSync, readFileSync } from "node:fs";
import { createServer, IncomingMessage, Server, ServerResponse } from "node:http";
import { basename, resolve } from "node:path";
import { runFiles, withDefaults } from "../run.ts";
import { LogReader } from "./log.ts";
import { gameStream, readGenerationBest, sampleFixtures } from "./replay.ts";

const page = readFileSync(new URL("./page.html", import.meta.url));

export interface DashboardOptions {
  host?: string;
  port?: number;
  /** How often the log is looked at, in ms. */
  pollMs?: number;
  /** Ticks a second the page plays the sample games at (default 8, the room's). */
  speed?: number;
  /** The most games shown for a generation (default 6). */
  games?: number;
}

export interface Dashboard {
  server: Server;
  /** The address it listens on, once `listen` is done. */
  url: string;
  close(): Promise<void>;
}

/** A run folder's settings, or an error saying why it isn't a run. Only reads. */
export function readRunSettings(run: string): Record<string, unknown> {
  const { settings } = runFiles(run);
  if (!existsSync(settings)) throw new Error(`${run} isn't a run folder: it has no settings.json`);
  try {
    const parsed = JSON.parse(readFileSync(settings, "utf8"));
    if (!parsed || typeof parsed !== "object" || typeof parsed.personality !== "string") throw new Error("not a run's settings");
    return parsed;
  } catch (error) {
    throw new Error(`${run} isn't a run folder: ${settings} is unusable (${(error as Error).message})`);
  }
}

const send = (res: ServerResponse, event: string, data: unknown) => res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);

/**
 * Serves a run folder: the page, its settings, and the log as Server-Sent Events (the whole log on
 * connect, then one `generation` event per new line; `log` again if lines were rewritten). It only ever reads the folder.
 */
export async function startDashboard(run: string, options: DashboardOptions = {}): Promise<Dashboard> {
  const settings = readRunSettings(run);
  const speed = options.speed ?? 8;
  const gameLimit = options.games ?? 6;
  const reader = new LogReader(runFiles(run).log);
  reader.poll();
  const clients = new Set<ServerResponse>();
  const status = () => ({ now: Date.now(), updatedAt: reader.updatedAt });

  const tick = () => {
    const change = reader.poll();
    if (!change) return;
    for (const res of clients) {
      if (change.kind === "reset") send(res, "log", { lines: change.lines, ...status() });
      else for (const line of change.lines) send(res, "generation", { line, ...status() });
    }
  };
  const timer = setInterval(tick, options.pollMs ?? 500);
  const beat = setInterval(() => { for (const res of clients) send(res, "status", status()); }, 5000);

  /** Generation g's sample games as Server-Sent Events: `generation`, then each game's `start`, `tick`s and `end`, then `done`. */
  const games = async (req: IncomingMessage, res: ServerResponse) => {
    const g = Number(new URL(req.url ?? "/", "http://localhost").searchParams.get("generation"));
    const brain = Number.isInteger(g) && g >= 0 ? readGenerationBest(run, g) : undefined;
    if (!brain) {
      res.writeHead(404, { "content-type": "text/plain" }).end(`no generation ${g} saved in this run\n`);
      return;
    }
    res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache", connection: "keep-alive" });
    let open = true;
    req.on("close", () => { open = false; });
    const full = withDefaults(settings as never);
    const sample = sampleFixtures(full, g, gameLimit);
    send(res, "generation", { generation: g, games: sample.length });
    for (const { fixture, index } of sample) {
      for (const message of gameStream(fixture, index, brain, full.fps)) {
        if (!open) return;
        send(res, message.kind, message);
      }
      await new Promise((resolve) => setImmediate(resolve)); // let the trainer's box and other requests in between games
    }
    if (open) { send(res, "done", {}); res.end(); }
  };

  const handle = (req: IncomingMessage, res: ServerResponse) => {
    const path = new URL(req.url ?? "/", "http://localhost").pathname;
    if (req.method !== "GET") {
      res.writeHead(405, { allow: "GET" }).end();
    } else if (path === "/") {
      res.writeHead(200, { "content-type": "text/html; charset=utf-8" }).end(page);
    } else if (path === "/api/run") {
      res.writeHead(200, { "content-type": "application/json" }).end(JSON.stringify({ name: basename(resolve(run)), settings, replay: { speed, games: gameLimit } }));
    } else if (path === "/games") {
      games(req, res);
    } else if (path === "/events") {
      res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache", connection: "keep-alive" });
      tick(); // so this client's whole log is current, and nothing is sent twice
      send(res, "log", { lines: reader.all, ...status() });
      clients.add(res);
      req.on("close", () => clients.delete(res));
    } else {
      res.writeHead(404, { "content-type": "text/plain" }).end("not found\n");
    }
  };

  const server = createServer(handle);
  await new Promise<void>((done, fail) => {
    server.once("error", fail);
    server.listen(options.port ?? 8080, options.host ?? "127.0.0.1", done);
  }).catch((error) => {
    clearInterval(timer);
    clearInterval(beat);
    throw error;
  });
  const address = server.address();
  const url = typeof address === "object" && address ? `http://${address.address.includes(":") ? `[${address.address}]` : address.address}:${address.port}` : "";
  return {
    server,
    url,
    close: () => new Promise<void>((done) => {
      clearInterval(timer);
      clearInterval(beat);
      for (const res of clients) res.end();
      server.close(() => done());
      server.closeAllConnections();
    })
  };
}
