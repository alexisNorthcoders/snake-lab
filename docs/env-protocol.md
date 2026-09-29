# The `env` protocol

`npm run env` (or `npx tsx src/cli/env.ts`) runs many matches at once for a learner in another process. The
learner sends actions over the process's **stdin** and reads observations and facts from its **stdout**.
Anything else the process prints goes to stderr. Rules stay in the TypeScript engine; nothing here is a copy.

## Frames

Every message, both ways, is one frame. All integers and floats are **little-endian**.

| Bytes | Field |
| --- | --- |
| 0-3 | `jsonLength`, uint32 |
| 4-7 | `payloadLength`, uint32 |
| 8 to 8+`jsonLength` | the header: a UTF-8 JSON object |
| then `payloadLength` bytes | the payload (may be empty) |

The process answers each request with exactly one response, in order. It handles one request at a time, so a
client may pipeline but needn't. `protocol` is 1; a reader should refuse another.

## Requests

### `reset`: start a batch

```json
{ "type": "reset", "seed": 5, "encoder": 2, "fps": 8, "endOnDeath": true,
  "matches": [ { "opponents": ["rookie"], "mode": "timed", "learnerDelay": 2, "opponentDelay": 2, "seat": 0 } ] }
```

No payload. It replaces any batch already running. `matches` has one spec a match in the batch (at least one).

| Field | Default | Meaning |
| --- | --- | --- |
| `seed` | required | whole number in [0, 2^32). Everything the env draws comes from `mulberry32(seed)`: every match's seed, and whatever `mixed`, a delay range or `"random"` asks it to draw, in match order |
| `encoder` | 2 | 1 (23 values a match) or 2 (328). One for the whole batch, so every observation is the same size |
| `fps` | 8 | ticks a second: sets a timed round's length |
| `endOnDeath` | true | end a match's episode as soon as the learner dies, rather than playing on among the others |
| `opponents` | `[]` | 0 to 3 roster ids or brain file paths, one a seat besides the learner's (none is a snake alone on the board) |
| `mode` | `timed` | `timed`, `endless`, or `mixed` (drawn each match) |
| `learnerDelay`, `opponentDelay` | 2 | reaction delay in ticks: a number, or `[min, max]` (inclusive) drawn each match. The opponent one is drawn separately for each opponent |
| `seat` | 0 | the learner's seat (0 to `opponents.length`), or `"random"` (drawn each match). The opponents fill the other seats in order |

Each match's seed is the batch generator's next draw: `floor(rng() * 2^32)`. It's drawn first, then (if asked)
the mode, the seat, the learner's delay, then each opponent's delay, in that order. At a reset the matches are
started in index order; later, a match that ends is restarted in the middle of its step, in index order.

### `step`: one tick of every match

```json
{ "type": "step" }
```

Payload: `matches` bytes (uint8), one action a match in match order: **0 left, 1 straight, 2 right** of the way
the snake last moved. A dead learner's action is ignored. Anything but 0, 1 or 2, or the wrong count, is an
error and no match moves.

### `close`

`{ "type": "close" }`. Answered with `{ "type": "closed" }`, then the process exits 0. Closing stdin exits too.

## Responses

### `batch`: to `reset` and to `step`

Header:

```json
{ "type": "batch", "protocol": 1, "matches": 3, "obsSize": 328,
  "factFields": ["scoreGained", "ate", "kills", "alive", "deathCause", "deathBy", "ended", "outcome", "reason", "ticks", "seat"] }
```

Payload, in this order, with `n` = `matches` and `F` = `factFields.length` (11):

1. **Facts**: `n × F` int32. Match `i`'s field `k` is at int32 index `i × F + k`. Facts, not rewards.
2. **Observations**: `n × obsSize` float32. Match `i`'s are at float32 index `i × obsSize`. They begin at byte
   offset `n × F × 4`, a multiple of 4.

A **reset** answers with each match's first observation, and facts all zero except `alive` = 1 and `seat`.

A **step** answers with the facts of the tick just played. If a match's `ended` is 1, the facts are those of its
last tick, and its observation is **the first of the next match**, already started on a new seed. Otherwise
the observation is the learner's view after the tick.

The observation is the learner's view, with its reaction delay, through the encoder: exactly the package's
`encode` (v1) or `encodeV2` (v2) output, as float32.

### Facts

| Field | Meaning |
| --- | --- |
| `scoreGained` | the learner's score after the tick minus before it (negative if an endless round drained it) |
| `ate` | pellets it ate this tick |
| `kills` | snakes it killed this tick. A head-on collision kills for neither |
| `alive` | 1 if the learner is alive after the tick |
| `deathCause` | set on the tick it died: 0 none, 1 self, 2 body, 3 head-on, 4 starved |
| `deathBy` | who it ran into: the killer's seat number, or -1 (none: its own doing, starving or nobody's body). In a head-on it names the other snake, though that's no kill |
| `ended` | 1 if the match's episode ended on this tick (see below) |
| `outcome` | on the last tick: 1 won, 2 lost, 3 drawn; 0 before |
| `reason` | on the last tick, why it ended: 1 last-standing, 2 time-up, 3 learner-died (see below); 0 before |
| `ticks` | ticks played in the match so far |
| `seat` | the learner's seat in the match these facts describe (after an `ended`, the match that just ended) |

An episode ends when the engine ends the round (`reason` 1 or 2; `outcome` is won if the learner is the winner,
drawn if the round has none, lost otherwise). A learner alone on the board that dies has no winner: `drawn`
(read `alive` and `deathCause`). With `endOnDeath` (the default), an episode also ends the tick the learner dies while
others live on: `reason` 3, `outcome` lost. If the round ends on that same tick, the engine's result stands.

### `error`

`{ "type": "error", "message": "…" }`, no payload. Sent for a bad message: an unknown `type`, a bad config, an
opponent that isn't in the roster or a valid brain file, `step` before `reset`, or bad actions. The batch, if there
is one, is left as it was, and the process carries on.

## Determinism

The same `reset` and the same actions give the same bytes, response by response. Nothing reads the clock or
`Math.random`. Opponent brain files are read once at `reset`.
