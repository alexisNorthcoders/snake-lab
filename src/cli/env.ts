import { EnvSession, FrameReader, encodeFrame } from "../env.ts";

const usage = `Runs many matches at once for a learner in another process, over stdin and stdout.

Usage: npm run env   (or: npx tsx src/cli/env.ts)

Takes no options: the protocol is in docs/env-protocol.md. Anything but frames goes to stderr.`;

if (process.argv.includes("--help") || process.argv.includes("-h")) {
  console.log(usage);
  process.exit(0);
}
if (process.argv.length > 2) {
  console.error(usage);
  process.exit(2);
}

const session = new EnvSession();
const reader = new FrameReader();

process.stdin.on("data", (chunk: Buffer) => {
  let frames;
  try {
    frames = reader.push(chunk);
  } catch (error) {
    console.error(`bad frame: ${(error as Error).message}`);
    process.exit(1);
  }
  for (const frame of frames) {
    process.stdout.write(encodeFrame(session.handle(frame)));
    if (session.closed) process.exit(0);
  }
});
process.stdin.on("end", () => process.exit(0));
