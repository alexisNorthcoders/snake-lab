# `playMatch` rebuilt on the steppable match

- **Date:** 2026-09-29
- **Engine and bots:** `snake-colyseus` at `engine-v4.2.0` (both before and after)
- **Machine:** Raspberry Pi 5, Node 24.15
- **Command:** `npm run bench -- --matches 300` (the defaults otherwise: rookie against rookie, timed, 8 fps, delay 2)

| | Ticks a second | Matches a second |
| --- | ---: | ---: |
| Before (`playMatch` with its own loop), two runs | 911, 912 | 13.1 |
| After (`playMatch` stepping a `SteppedMatch`), two runs | 913, 911 | 13.1 |

No slower. Not measured here: the `env` subprocess's own throughput (run it from the Python trainer).
