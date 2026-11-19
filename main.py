"""gta6-trophy-rarity-forecast-01 entry: per-trophy logit-trend forecast.

Forecasts every GTA VI (PS5) trophy's earn rate on Exophase at the case's
target, as five quantiles (5/25/50/75/95%), scored by pinball loss on the
logit scale.

Model, per trophy:
  * Fit a least-squares line of logit(rate) against ln(1 + hours since
    2026-11-19T00:00:00Z) over the case's last 24 hours of reads. Launch
    night's reads are all identical, so the fit is flat there and the
    forecast falls back to the last read, as it should.
  * Shrink each trophy's slope toward its tier's mean slope, so a noisy
    single-trophy line cannot run away over a 4.5-day horizon.
  * Extrapolate the shrunk line to the target, then clamp the median rate
    to a cap set by what the trophy asks of the player: story/mission
    trophies rise fastest, the platinum stays near zero, and grind /
    collectible trophies (100% completion, 250 stunts, card sets, 100
    locations, ...) stay low. Caps are read off the trophy's description,
    not its id, so they still apply if the list ever changes.
  * Stale-page dilution: Exophase refreshes its counts on its own
    schedule, and a refresh floods the denominator with newer, less
    progressed players, so most rates fall at a refresh. If the page has
    not changed for hours, the next refresh is overdue and the median is
    shifted down in logit space, scaled by how stale the page is.
  * Spread: honest and wide for a multi-day horizon -- a base term for
    refresh/step risk, a random-walk term growing with sqrt(days), and
    the fit's residual -- all wider than the house baseline's
    0.5*sqrt(days), which the launch-night refresh blew straight through
    (practice loss about 1.51, 90% coverage about 0).

Deterministic: pure float arithmetic, fixed rounding, files processed in
name order, no randomness, no network, standard library only.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Launch night, the origin of the log-time axis.
EPOCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# Spread (logit units): base refresh/step risk, random walk per sqrt day.
BASE_SPREAD = 0.6
RW_SPREAD = 0.7
# How far the median may move from the last read, in logit units.
MAX_LOGIT_MOVE = 3.5
# Stale-page dilution: full shift once the page is 6h stale, none under 1h.
STALE_FULL_HOURS = 6.0
STALE_MIN_HOURS = 1.0
STALE_SHIFT = 0.5
# Slope shrinkage: weight on a trophy's own slope, n reads in the fit.
SHRINK_N = 8.0
# Only reads this recent feed the trend fit.
FIT_WINDOW_HOURS = 24.0

# Median-rate caps (percent) by what the trophy asks of the player.
PLATINUM_CAP = 2.0
GRIND_CAP = {"gold": 1.5, "silver": 8.0, "bronze": 6.0}
STORY_CAP = {"gold": 15.0, "silver": 25.0, "bronze": 70.0}
MISC_CAP = {"gold": 12.0, "silver": 15.0, "bronze": 30.0}

_GRIND_PATTERNS = [
    "100% completion", "250 stunts", "100 locations", "50 challenges",
    "collectible card", "gold medal in all", "8 mission strands",
    "5 bank vaults", "laundering", "all garage properties",
    "20 different races", "discover 100 vehicles", "every spot",
    "every location", "register 10 vehicles", "earn a gold medal",
]
_STORY_RE = re.compile(r"^complete (chapter|the final mission|gone fishing)",
                       re.IGNORECASE)


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    value = float(Decimal(repr(x)).quantize(Decimal("0.0001"),
                                             rounding=ROUND_HALF_UP))
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def log_hours(at: datetime) -> float:
    hours = (at - EPOCH).total_seconds() / 3600.0
    return math.log1p(max(0.0, hours))


def category(trophy: dict) -> str:
    """What the trophy asks of the player, from its description."""
    if trophy["tier"] == "platinum":
        return "platinum"
    desc = (trophy.get("description") or "").lower()
    if any(p in desc for p in _GRIND_PATTERNS):
        return "grind"
    if _STORY_RE.match(trophy.get("description") or ""):
        return "story"
    if desc.startswith("complete "):
        return "story"
    return "misc"


def cap_for(trophy: dict) -> float:
    kind = category(trophy)
    if kind == "platinum":
        return PLATINUM_CAP
    table = {"grind": GRIND_CAP, "story": STORY_CAP, "misc": MISC_CAP}[kind]
    return table.get(trophy["tier"], 30.0)


def fit_line(points):
    """Least squares y = a + b*x; returns (a, b, residual_sd) or None."""
    n = len(points)
    if n < 3:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in points)
    b = sxy / sxx
    a = my - b * mx
    resid = math.sqrt(sum((y - (a + b * x)) ** 2 for x, y in points) / n)
    return a, b, resid


def stale_hours(case: dict) -> float:
    """Hours since the page last changed (rates or players tracked)."""
    history = case["history"]
    if len(history) < 2:
        return 0.0
    last = history[-1]
    checkpoint = parse_time(case["at"])
    for read in reversed(history[:-1]):
        if read.get("players_tracked") != last.get("players_tracked") \
                or read.get("rates") != last.get("rates"):
            return max(0.0, (checkpoint - parse_time(read["at"])).total_seconds() / 3600.0)
    first = parse_time(history[0]["at"])
    return max(0.0, (checkpoint - first).total_seconds() / 3600.0)


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = case["horizon_minutes"] / 1440.0
    x_target = log_hours(parse_time(case["target"]))
    checkpoint = parse_time(case["at"])

    # Last non-null rate per trophy, for the fallback level.
    last_rate = []
    for i in range(len(trophies)):
        rate = None
        for read in reversed(history):
            r = read["rates"][i]
            if r is not None:
                rate = r
                break
        last_rate.append(rate)

    # Per-trophy trend fits over the last 24h of reads.
    window_start = checkpoint.timestamp() - FIT_WINDOW_HOURS * 3600.0
    fits = []
    for i in range(len(trophies)):
        points = []
        for read in history:
            if parse_time(read["at"]).timestamp() < window_start:
                continue
            r = read["rates"][i]
            if r is None:
                continue
            points.append((log_hours(parse_time(read["at"])), logit(r)))
        if len(points) < 3:
            points = []
            for read in history:
                r = read["rates"][i]
                if r is None:
                    continue
                points.append((log_hours(parse_time(read["at"])), logit(r)))
        fits.append(fit_line(points))

    # Tier mean slopes, for shrinkage.
    tier_slopes = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        slopes = [f[1] for t, f in zip(trophies, fits)
                  if t["tier"] == tier and f is not None]
        tier_slopes[tier] = sum(slopes) / len(slopes) if slopes else 0.0

    stale = stale_hours(case)
    stale_shift = 0.0
    if stale > STALE_MIN_HOURS:
        frac = min(1.0, (stale - STALE_MIN_HOURS)
                   / (STALE_FULL_HOURS - STALE_MIN_HOURS))
        stale_shift = STALE_SHIFT * frac

    rows = []
    for i, trophy in enumerate(trophies):
        base = last_rate[i]
        if base is None:
            base = {"platinum": 0.5, "gold": 1.0,
                    "silver": 1.0, "bronze": 1.0}[trophy["tier"]]
        last_lg = logit(base)
        fit = fits[i]
        if fit is None:
            med_lg = last_lg
            resid = 0.0
        else:
            a, b, resid = fit
            n = max(3, sum(1 for read in history
                           if read["rates"][i] is not None))
            w = n / (n + SHRINK_N)
            slope = w * b + (1.0 - w) * tier_slopes[trophy["tier"]]
            med_lg = a + slope * x_target
            med_lg = min(last_lg + MAX_LOGIT_MOVE,
                         max(last_lg - MAX_LOGIT_MOVE, med_lg))
        med_lg -= stale_shift
        med = inv_logit(med_lg)
        # The cap limits growth above the last read; it never drags a
        # trophy below where it already stands (trophy 2 sits at 100%
        # and stayed there through the first refresh).
        if med > base:
            med = min(med, cap_for(trophy))
        med = clamp_pct(med)
        med_lg = logit(med)
        spread = math.sqrt(BASE_SPREAD ** 2
                           + (RW_SPREAD * math.sqrt(horizon_days)) ** 2
                           + min(resid, 1.0) ** 2)
        rows.append({"id": trophy["id"],
                     "q": [round4(inv_logit(med_lg + z * spread))
                           for z in Z]})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
