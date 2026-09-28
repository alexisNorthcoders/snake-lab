# First gauntlet benchmarks

- **Date:** 2026-09-28
- **Engine and bots:** `snake-colyseus` at `engine-v3.1.0`
- **Machine:** Raspberry Pi 5 (4 cores; the gauntlet uses one), Node 24.15
- **Commands:** `npm run gauntlet -- rookie` and `npm run gauntlet -- dummy`, at the defaults:
  base seed 1, 100 seeds per cell, each played in both seats (200 matches a cell), 8 fps.

Both fail the bar, as they should: 60% of matches won against the rookie at delay 2, both modes together.

| Candidate | Matches | Time | Won against the rookie at delay 2 | Verdict |
| --- | ---: | ---: | ---: | --- |
| Rookie | 4000 | 284 s | 8.0% | fail (exit 1) |
| Dummy | 2000 | 101 s | 4.0% | fail (exit 1) |

The time is slow but workable: about 14 matches a second, so a full gauntlet against the two-snake
roster takes under 5 minutes. Each extra roster snake adds about 2.5 minutes.

## What the numbers show

- **Rookie against rookie wins far less than half.** The seat swap makes its wins and losses equal
  in every cell, as they should be. But at delay 1 and above, most matches are draws (84% at delay 2):
  the two snakes meet head-on, often within 7 to 18 ticks, and both die
  on the same tick. Seeing the other snake a tick or more late, neither turns away in time. At
  delay 0 there are no draws and the split is 50/50.
- **Timed and endless give the same numbers in every cell.** Nearly every match ends with a death
  long before a timed round's tick limit, so the limit never comes into play. The modes will only
  differ for snakes that survive longer than these do.
- **Dummy loses to the rookie at every delay**, and mostly draws at delay 1 and above, for the same
  reason: head-on collisions.

## Rookie

```
Rookie: seeds 1 to 100, each in both seats, at 8 fps
4000 matches in 283.7s

  opponent     delay  mode     matches     win    loss    draw
  Rookie           0  timed        200   50.0%   50.0%    0.0%
  Rookie           0  endless      200   50.0%   50.0%    0.0%
  Rookie           1  timed        200    8.5%    8.5%   83.0%
  Rookie           1  endless      200    8.5%    8.5%   83.0%
  Rookie           2  timed        200    8.0%    8.0%   84.0%
  Rookie           2  endless      200    8.0%    8.0%   84.0%
  Rookie           3  timed        200   21.5%   21.5%   57.0%
  Rookie           3  endless      200   21.5%   21.5%   57.0%
  Rookie           4  timed        200   21.5%   21.5%   57.0%
  Rookie           4  endless      200   21.5%   21.5%   57.0%
  Dummy            0  timed        200   68.5%   30.5%    1.0%
  Dummy            0  endless      200   68.5%   30.5%    1.0%
  Dummy            1  timed        200   20.5%    4.5%   75.0%
  Dummy            1  endless      200   20.5%    4.5%   75.0%
  Dummy            2  timed        200   21.0%    4.0%   75.0%
  Dummy            2  endless      200   21.0%    4.0%   75.0%
  Dummy            3  timed        200   29.0%   14.0%   57.0%
  Dummy            3  endless      200   29.0%   14.0%   57.0%
  Dummy            4  timed        200   30.0%   13.5%   56.5%
  Dummy            4  endless      200   30.0%   13.5%   56.5%

Bar: win at least 60.0% against Rookie at delay 2, both modes together.
Rookie won 8.0%: FAIL
```

## Dummy

```
Dummy: seeds 1 to 100, each in both seats, at 8 fps
2000 matches in 100.6s

  opponent     delay  mode     matches     win    loss    draw
  Rookie           0  timed        200   30.5%   68.5%    1.0%
  Rookie           0  endless      200   30.5%   68.5%    1.0%
  Rookie           1  timed        200    4.5%   20.5%   75.0%
  Rookie           1  endless      200    4.5%   20.5%   75.0%
  Rookie           2  timed        200    4.0%   21.0%   75.0%
  Rookie           2  endless      200    4.0%   21.0%   75.0%
  Rookie           3  timed        200   14.0%   29.0%   57.0%
  Rookie           3  endless      200   14.0%   29.0%   57.0%
  Rookie           4  timed        200   13.5%   30.0%   56.5%
  Rookie           4  endless      200   13.5%   30.0%   56.5%

Bar: win at least 60.0% against Rookie at delay 2, both modes together.
Dummy won 4.0%: FAIL
```
