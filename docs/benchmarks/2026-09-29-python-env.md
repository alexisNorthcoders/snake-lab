# The Python environment's throughput

- **Date:** 2026-09-29
- **Engine and bots:** `snake-colyseus` at `engine-v4.2.0`
- **Machine:** Raspberry Pi 5, Node 24.15, Python 3.11.2
- **Command:** `python -m snake_lab.bench --seconds 3 --batches 1,4,16,64,256` (from `python/`; random actions, one rookie opponent, learner in a random seat, timed and endless rounds, delay 2), and again with `--encoder 1 --batches 16,64`. 3 seconds a batch size, one run each.

| Batch | v2 steps a second | v1 steps a second |
| ---: | ---: | ---: |
| 1 | 1219 | |
| 4 | 1417 | |
| 16 | 1466 | 1317 |
| 64 | 1490 | 1363 |
| 256 | 1570 | |

A step is one tick of one match (a batch of 64 is 64 steps). The engine subprocess is one Node thread, so it tops
out around 1,500 steps a second: the gain past a batch of 16 is small, and the per-call overhead (which dominates at
a batch of 1) is gone by then. **Best: 64 to 256; 64 is the pick**, since a bigger batch makes each PPO rollout wait
longer for little more throughput. Encoder v1 is about 8% faster than v2 (its observation is 14 times smaller).
To use more of the Pi's cores, run several environments (one subprocess each).
