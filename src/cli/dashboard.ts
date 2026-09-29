import { parseArgs } from "node:util";
import { startDashboard } from "../dashboard/server.ts";
import { exit, orExit, parseNumber } from "./args.ts";

const usage = `Serves a page that charts a training run: best and mean fitness per generation, the stages shaded,
growing as the run trains. It reads the run folder and never writes to it. It listens on 127.0.0.1
unless --host says otherwise: reach it from another machine with ssh -L, and never expose it publicly.

Usage: npm run dashboard -- <run> [options]

  <run>          the run folder
  --port <n>     the port (default 8080)
  --host <addr>  the address to listen on (default 127.0.0.1); 0.0.0.0 listens on the LAN`;

const parsed = orExit(usage, () => parseArgs({
  options: { port: { type: "string", default: "8080" }, host: { type: "string", default: "127.0.0.1" }, help: { type: "boolean" } },
  allowPositionals: true,
  strict: true
}));
if (parsed.values.help) exit(usage, 0);

const { run, port, host } = orExit(usage, () => {
  if (parsed.positionals.length !== 1) throw new Error(`give one run folder, not ${parsed.positionals.length} arguments`);
  return { run: parsed.positionals[0], port: parseNumber("port", parsed.values.port), host: parsed.values.host };
});

try {
  const dashboard = await startDashboard(run, { host, port });
  console.log(`Dashboard for ${run} at ${dashboard.url}`);
} catch (error) {
  exit(`Can't serve ${run}: ${(error as Error).message}`);
}
