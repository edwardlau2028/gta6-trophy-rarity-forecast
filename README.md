# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/edwardlau2028/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/edwardlau2028/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Forecasts every Grand Theft Auto VI (PlayStation 5) trophy's earn rate on
Exophase at the case's target, as five quantiles per trophy (5%, 25%, 50%,
75%, 95%), scored by pinball loss on the logit scale.

## How it works

`main.py` (Python 3.13, standard library only) reads every `*.in` case in
`$MAKEFUN_INPUT_DIR` (default `/input`) and writes one `<id>.out` forecast
per case to `$MAKEFUN_OUTPUT_DIR` (default `/output`).

Per trophy it:

1. Fits a least-squares line of `logit(rate)` against
   `ln(1 + hours since 2026-11-19T00:00:00Z)` over the last 24 hours of
   reads, and shrinks the slope toward the tier's mean slope.
2. Extrapolates the shrunk line to the target, limiting the total logit
   move and capping the median rate by what the trophy asks of the
   player: story/mission trophies rise fastest, the platinum stays near
   zero, grind/collectible trophies (100% completion, 250 stunts, card
   sets, 100 locations, ...) stay low. Caps are read off the trophy's
   description, and only ever limit growth above the last read.
3. Shifts the median down when the page is stale: Exophase refreshes on
   its own schedule, and a refresh floods the denominator with newer,
   less progressed players, so most rates fall at a refresh.
4. Spreads the quantiles honestly wide for a multi-day horizon: a base
   term for refresh/step risk, a random-walk term growing with
   `sqrt(days)`, and the trend fit's residual -- wider than the house
   baseline's `0.5*sqrt(days)`, which the launch-night refresh blew
   straight through.

Output is deterministic: two runs on the same case write byte-identical
files.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Files

- `main.py` -- the entry (the only file the grader runs).
- `README.md` -- this file.
- `LICENSE` -- MIT.
