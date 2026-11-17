"""Forecast GTA VI's launch-day Twitch viewers, hour by hour (round 2).

Reads every ``<id>.in`` case in the input directory (``$MAKEFUN_INPUT_DIR``,
else ``/input``) and writes one ``<id>.out`` forecast per case to the output
directory (``$MAKEFUN_OUTPUT_DIR``, else ``/output``). Python 3.13, standard
library only, no network, fully deterministic.

Why this model
---------------
A launch day's shape is governed by *how* the game unlocks, not just by how
big its first hour is:

* Console-only launches unlock at rolling midnight local time, so every one
  of them shares the same timezone rollout (New Zealand first, then Asia and
  Europe, the UK at 00:00 UTC, the US east coast at 05:00 UTC and the west
  coast at 08:00 UTC). In the kit's comparables their day curves are far more
  predictable than PC launches': the per-hour residual spread of log(v_h/v_1)
  around a log-p1 regression is 0.16-0.36 for console-only curves versus
  0.46-0.71 for curves that include a PC simultaneous global unlock (whose
  unlock hour varies from game to game). GTA VI is console-only (PS5 and Xbox
  Series X|S), as are all four practice launches, so the forecast leans on
  the console analogues.
* Within the already-unlocked regime (hour-1 print >= 5,000) the log-ratio
  log(v_h/v_1) is close to linear in log(p1) with a slope that grows from ~0
  at hour 2 to about -0.5 at hour 20: bigger first hours saturate, smaller
  ones still ramp. A pooled linear fit over the large-print curves beats a
  local kernel there (LOO 0.258 vs 0.269), while the kernel is better for the
  tiny-print regime, where launches explode by orders of magnitude once they
  unlock. The shipped forecast is a smooth hybrid of the two, switching
  regimes around p1 = 5,000.
* The evening before release already carries a little signal: the mean
  viewership over the six hourly points before midnight, relative to the
  first-hour print, trims the residual spread further (about 0.63 -> 0.58 at
  hour 15). It enters as a second regressor with its coefficient shrunk
  halfway to zero, so a noisy pre-release read cannot dominate.

Method (per case)
-----------------
1. Curves: past launches from ``comparables`` whose hourly series span a
   release window (~17 days, not the months-long pre-release history of the
   target's own category). Each curve records its hour-1 print, its day
   curve, whether it is console-only (from its ``platforms``), and its
   pre-midnight evening level.
2. Regime A (large prints): per hour, a weighted least-squares fit of
   log(v_h/v_1) on log(p1) and the pre-evening feature, over curves with
   p1 >= 5,000, console-only curves weighted x3. The fitted median is
   smoothed over neighbouring hours. The spread is the residual spread of
   the target's own group (console-only curves when the target is one,
   otherwise the PC curves), corrected for the fitted degrees of freedom,
   smoothed over hours and floored at 0.25 in log units. Quantiles are
   lognormal around the fitted median.
3. Regime B (small prints): the round-1 kernel forecast -- Gaussian-kernel
   weighted empirical quantiles of the log-ratios and of the absolute hourly
   levels, blended in log space with a weight that rises with p1.
4. The two regimes are log-blended with a logistic weight centred at
   p1 = 5,000, so a case with a large first hour is essentially regime A and
   a case with a tiny one is essentially regime B.
5. Whether the target is console-only is read from the case itself: the
   comparable carrying the target game's own pre-release history states its
   platforms, and the four practice launches are known console-only
   releases. Anything else falls back to the pooled (PC) spread.

Local validation (leave-one-out over the 37 usable comparable curves, run
through the shipped code): hybrid loss 0.381 overall, 0.244 for p1 >= 5,000
and 0.147 for console-only large curves, versus the round-1 model's 0.385 /
0.269 / 0.219 on the same protocol. Quantiles are clamped to ``(0, 1e9]``, forced
non-decreasing, and rounded to 2 decimals. No randomness anywhere, so two
runs are byte-identical.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- regime A (large prints) ---------------------------------------------
LARGE_MIN_P1 = 5000.0     # curves below this do not train regime A
CONSOLE_WEIGHT = 3.0      # console-only analogues count triple
PRE_SHRINK = 0.5          # pre-evening coefficient shrunk halfway to zero
SIGMA_FLOOR = 0.25        # log units; console residual spread is ~0.2-0.35
MIN_ROWS_PER_HOUR = 5

# --- regime B (small prints, round-1 kernel) ------------------------------
KERNEL_BW = 1.1           # Gaussian kernel bandwidth on log(p1)
BLEND_LOW = 0.40          # ratio-side weight for a tiny first hour
BLEND_HIGH = 0.85         # ratio-side weight for a large first hour
BLEND_CENTER = math.log(6000.0)
BLEND_SLOPE = 0.7

# --- hybrid switch ---------------------------------------------------------
HYBRID_CENTER = 5000.0   # p1 where regimes A and B weigh equally
HYBRID_WIDTH = 0.3        # logistic width in log units (sharp switch)

# --- curve extraction ------------------------------------------------------
MAX_CURVE_SPAN_HOURS = 500.0
MIN_CURVE_HOURS = 20
FALLBACK_SIGMA = 0.30

# Launches known to be console-only (the target game is read from the case;
# the four practice launches are public console-only releases; the other
# names are the console-only launches among the kit's comparables, listed
# so the console branch is exercised the same way in validation as in
# deployment -- all of them are public platform facts, not fitted values).
CONSOLE_ONLY_NAMES = (
    "grand theft auto vi",
    "gta vi",
    "wolverine",
    "red dead redemption 2",
    "god of war ragnar",
    "spider-man 2",
    "spiderman 2",
    "death stranding 2",
    "final fantasy vii rebirth",
    "ghost of tsushima",
    "ghost of yotei",
    "saros",
    "stellar blade",
    "the last of us part ii",
)


def _parse_time(text):
    """Parse an ISO-8601 UTC timestamp into epoch seconds (float)."""
    try:
        dt = datetime.fromisoformat(str(text).strip().replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _is_console_only(platforms):
    """A launch is console-only when no PC or Switch platform is listed."""
    text = str(platforms or "").lower()
    return ("pc" not in text) and ("switch" not in text)


def _hour_index(at, day_start):
    """Hour number relative to day_start: 1 is the first hour of the day,
    0 the hour before it, -1 the one before that, and so on."""
    return int((at - day_start) // 3600) + 1


def extract_curves(case):
    """Release-day curves from comparables.

    Returns a list of dicts, one per usable launch, sorted by name:
    ``p1`` (hour-1 viewers), ``by_hour`` (hour -> viewers for hours 1..24),
    ``console`` (console-only launch or not), ``pre`` (log of the mean
    viewership over the six hourly points before midnight, divided by p1,
    or None when those points are missing).
    """
    curves = []
    for comp in sorted(case.get("comparables") or [],
                       key=lambda c: str(c.get("name", ""))):
        try:
            hourly = comp.get("hourly") or []
            if len(hourly) < 100:
                continue
            first = _parse_time(hourly[0].get("from"))
            last = _parse_time(hourly[-1].get("from"))
            if first is None or last is None:
                continue
            if (last - first) / 3600.0 > MAX_CURVE_SPAN_HOURS:
                continue
            day_start = _parse_time(comp.get("day_start"))
            if day_start is None:
                continue
            by_hour = {}
            pre_values = []
            for row in hourly:
                try:
                    viewers = row.get("viewers")
                    if not viewers or viewers <= 0:
                        continue
                    at = _parse_time(row.get("from"))
                    if at is None:
                        continue
                    h = _hour_index(at, day_start)
                    if 1 <= h <= 24:
                        by_hour[h] = float(viewers)
                    elif -6 <= h <= -1:
                        pre_values.append(float(viewers))
                except Exception:
                    continue
            if 1 in by_hour and len(by_hour) >= MIN_CURVE_HOURS:
                p1 = by_hour[1]
                pre = None
                if len(pre_values) >= 4:
                    pre = math.log((sum(pre_values) / len(pre_values)) / p1)
                curves.append({
                    "name": str(comp.get("name", "")),
                    "p1": p1,
                    "by_hour": by_hour,
                    "console": _is_console_only(comp.get("platforms", "")),
                    "pre": pre,
                })
        except Exception:
            continue
    return curves


def target_is_console(case, curves):
    """Is the case's own game a console-only launch?

    First, the comparable carrying the target game's own pre-release history
    states its platforms. Otherwise fall back to the built-in table of known
    console-only launches (the four practice games). Anything unknown is
    treated as not console-only, i.e. the pooled spread is used.
    """
    name = str((case.get("game") or {}).get("name", "")).strip().lower()
    if name:
        for comp in case.get("comparables") or []:
            comp_name = str(comp.get("name", "")).lower()
            if name and (name in comp_name or comp_name.startswith(name)):
                return _is_console_only(comp.get("platforms", ""))
        for known in CONSOLE_ONLY_NAMES:
            if known in name:
                return True
    return False


def target_pre_feature(case, p1):
    """Log of (mean viewers over the six hourly points before midnight) / p1,
    from the case's own ``history_hourly``; None when unavailable."""
    day_start = _parse_time(case.get("day_start"))
    if day_start is None:
        return None
    values = []
    for row in case.get("history_hourly") or []:
        try:
            viewers = row.get("viewers")
            if not viewers or viewers <= 0:
                continue
            at = _parse_time(row.get("from"))
            if at is None:
                continue
            if -6 <= _hour_index(at, day_start) <= -1:
                values.append(float(viewers))
        except Exception:
            continue
    if len(values) >= 4:
        return math.log((sum(values) / len(values)) / max(p1, 1.0))
    return None


def weighted_quantile(pairs, level):
    """Linearly interpolated weighted quantile of [(value, weight), ...]."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    if total <= 0 or not ordered:
        return ordered[len(ordered) // 2][0] if ordered else 0.0
    target = level * total
    acc = 0.0
    prev_value = ordered[0][0]
    for value, weight in ordered:
        if weight <= 0:
            continue
        acc += weight
        if acc >= target:
            if acc - weight < target:
                frac = (target - (acc - weight)) / weight
                return prev_value + frac * (value - prev_value)
            return value
        prev_value = value
    return ordered[-1][0]


def _weighted_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b)."""
    total = sum(ws)
    if total <= 0:
        return 0.0, 0.0
    mx = sum(w * x for w, x in zip(ws, xs)) / total
    my = sum(w * y for w, y in zip(ws, ys)) / total
    den = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if den <= 1e-12:
        return my, 0.0
    b = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys)) / den
    return my - b * mx, b


def _solve_3x3(matrix, rhs):
    """Solve a 3x3 linear system by Gaussian elimination; None if singular."""
    rows = [list(matrix[i]) + [rhs[i]] for i in range(3)]
    for i in range(3):
        piv = max(range(i, 3), key=lambda r: abs(rows[r][i]))
        rows[i], rows[piv] = rows[piv], rows[i]
        if abs(rows[i][i]) < 1e-12:
            return None
        for r in range(i + 1, 3):
            f = rows[r][i] / rows[i][i]
            for k in range(i, 4):
                rows[r][k] -= f * rows[i][k]
    beta = [0.0, 0.0, 0.0]
    for i in (2, 1, 0):
        beta[i] = (rows[i][3] - sum(rows[i][j] * beta[j]
                                    for j in range(i + 1, 3))) / rows[i][i]
    return beta


def _weighted_plane(xs, fs, ys, ws):
    """Weighted least squares y ~ b0 + b1*x + b2*f. Returns (b0, b1, b2)."""
    design = [[1.0, x, f] for x, f in zip(xs, fs)]
    normal = [[sum(w * r[i] * r[j] for w, r in zip(ws, design))
               for j in range(3)] for i in range(3)]
    target = [sum(w * r[i] * y for w, r, y in zip(ws, design, ys))
              for i in range(3)]
    return _solve_3x3(normal, target)


def _smooth(values):
    """Triangular smoothing over neighbouring hours (0.25, 0.5, 0.25)."""
    out = {}
    for h, v in values.items():
        acc = 0.0
        weight = 0.0
        for dh, w in ((-1, 0.25), (0, 0.5), (1, 0.25)):
            if h + dh in values:
                acc += w * values[h + dh]
                weight += w
        out[h] = acc / weight if weight else v
    return out


def _resid_sd(values, n_params=2):
    """Standard deviation with a degrees-of-freedom correction for a fit
    with ``n_params`` parameters; 0.7 when too few residuals to trust."""
    if len(values) < 3:
        return 0.7
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(var) * math.sqrt(len(values) / max(len(values) - n_params, 1))


def regime_a(p1, curves, console, pre_feat):
    """Large-print regime: weighted log-linear fit per hour.

    Returns {hour: [q05, q25, q50, q75, q95]} or None when too few large
    curves exist to fit.
    """
    train = [c for c in curves if c["p1"] >= LARGE_MIN_P1]
    if len(train) < MIN_ROWS_PER_HOUR:
        return None
    log_p1 = math.log(max(p1, 1.0))
    medians = {}
    sigmas = {}
    for h in range(2, 25):
        rows = [c for c in train if h in c["by_hour"]]
        if len(rows) < MIN_ROWS_PER_HOUR:
            medians[h] = 0.0
            sigmas[h] = 0.7
            continue
        xs = [math.log(c["p1"]) for c in rows]
        ys = [math.log(c["by_hour"][h] / c["by_hour"][1]) for c in rows]
        ws = [CONSOLE_WEIGHT if c["console"] else 1.0 for c in rows]
        use_pre = (pre_feat is not None
                   and all(c["pre"] is not None for c in rows))
        if use_pre:
            beta = _weighted_plane(
                xs, [c["pre"] for c in rows], ys, ws)
            if beta is not None:
                # Shrink the pre-evening coefficient halfway to zero: the
                # feature helps, but it must not dominate the fit.
                beta = (beta[0], beta[1], beta[2] * PRE_SHRINK)
                medians[h] = beta[0] + beta[1] * log_p1 + beta[2] * pre_feat
                resid = [y - (beta[0] + beta[1] * x + beta[2] * f)
                         for x, f, y in zip(xs, [c["pre"] for c in rows], ys)]
            else:
                use_pre = False
        if not use_pre:
            a, b = _weighted_line(xs, ys, ws)
            medians[h] = a + b * log_p1
            resid = [y - (a + b * x) for x, y in zip(xs, ys)]
        own = [r for r, c in zip(resid, rows)
               if c["console"] == console]
        other = [r for r, c in zip(resid, rows)
                 if c["console"] != console]
        # The spread of the target's own group; fall back to the pooled
        # residuals when the group is too small to estimate.
        sigmas[h] = _resid_sd(own) if len(own) >= 3 else _resid_sd(resid)
        if len(other) >= 3 and len(own) < 3:
            sigmas[h] = _resid_sd(other)
    medians = _smooth(medians)
    sigmas = _smooth(sigmas)
    out = {}
    for h in range(2, 25):
        sigma = max(SIGMA_FLOOR, sigmas[h])
        out[h] = [math.exp(log_p1 + medians[h] + z * sigma) for z in Z]
    return out


def _blend_weight(p1):
    """Weight on the ratio-based forecast (0.40 .. 0.85), smooth in log p1."""
    x = (math.log(max(p1, 1.0)) - BLEND_CENTER) / BLEND_SLOPE
    if x > 30:
        sig = 1.0
    elif x < -30:
        sig = 0.0
    else:
        sig = 1.0 / (1.0 + math.exp(-x))
    return BLEND_LOW + (BLEND_HIGH - BLEND_LOW) * sig


def regime_b(p1, curves):
    """Small-print regime: the round-1 kernel forecast.

    Gaussian-kernel weighted empirical quantiles of the log-ratios and of
    the absolute hourly levels, blended in log space. Kept because the
    log-linear fit of regime A extrapolates wildly for tiny first hours,
    where launches explode by orders of magnitude once they unlock.
    """
    p1 = max(1.0, float(p1))
    log_p1 = math.log(p1)
    alpha = _blend_weight(p1)
    out = {}
    for h in range(2, 25):
        ratio_pairs = []
        abs_pairs = []
        for c in curves:
            if h not in c["by_hour"]:
                continue
            w = math.exp(-0.5 * ((math.log(c["p1"]) - log_p1) / KERNEL_BW) ** 2)
            ratio_pairs.append((math.log(c["by_hour"][h] / c["by_hour"][1]), w))
            abs_pairs.append((math.log(c["by_hour"][h]), w))
        if ratio_pairs:
            qs = []
            for level in LEVELS:
                log_ratio = weighted_quantile(ratio_pairs, level)
                log_abs = weighted_quantile(abs_pairs, level)
                qs.append(math.exp(alpha * (log_p1 + log_ratio)
                                   + (1.0 - alpha) * log_abs))
        else:
            qs = [p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(h - 1))
                  for z in Z]
        out[h] = qs
    return out


def round2(x):
    """Round to 2 decimals, half up; whole numbers come back as int."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if value == int(value) else value


def forecast_hours(p1, curves, console, pre_feat):
    """Return {hour: [q05, q25, q50, q75, q95]} for hours 2..24."""
    p1 = max(1.0, float(p1))
    fa = regime_a(p1, curves, console, pre_feat)
    fb = regime_b(p1, curves)
    if fa is None:
        blended = fb
    else:
        x = (math.log(p1) - math.log(HYBRID_CENTER)) / HYBRID_WIDTH
        if x > 30:
            w = 1.0
        elif x < -30:
            w = 0.0
        else:
            w = 1.0 / (1.0 + math.exp(-x))
        blended = {}
        for h in range(2, 25):
            blended[h] = [math.exp(w * math.log(max(fa[h][i], 1.0))
                                   + (1.0 - w) * math.log(max(fb[h][i], 1.0)))
                          for i in range(5)]
    out = {}
    for h in range(2, 25):
        qs = sorted(min(max(q, 1.0), 1e9) for q in blended[h])
        out[h] = [round2(q) for q in qs]
    return out


def hour1_print(case):
    """Hour-1 print from `known`, falling back to the history's last point."""
    for row in case.get("known") or []:
        if row.get("hour") == 1:
            try:
                v = float(row.get("viewers"))
                if v > 0:
                    return v
            except Exception:
                pass
    history = case.get("history_hourly") or case.get("history") or []
    if history:
        try:
            v = float(history[-1].get("viewers"))
            if v > 0:
                return v
        except Exception:
            pass
    return 1.0


def forecast_case(case):
    p1 = hour1_print(case)
    curves = extract_curves(case)
    console = target_is_console(case, curves)
    pre_feat = target_pre_feature(case, p1)
    hours = forecast_hours(p1, curves, console, pre_feat)
    rows = []
    for row in sorted(case.get("forecast") or [], key=lambda r: r.get("hour", 0)):
        h = int(row["hour"])
        if 2 <= h <= 24:
            rows.append({"hour": h, "q": hours[h]})
    if len(rows) != 23:  # forecast list missing/malformed: fill hours 2..24
        rows = [{"hour": h, "q": hours[h]} for h in range(2, 25)]
    return {"hours": rows}


def persistence_forecast(p1):
    """Valid fallback forecast: every hour at p1 with a widening spread."""
    p1 = max(1.0, float(p1))
    return {
        "hours": [
            {
                "hour": h,
                "q": [
                    round2(p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(h - 1)))
                    for z in Z
                ],
            }
            for h in range(2, 25)
        ]
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        forecast = None
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except Exception:
            forecast = None
        if forecast is None:
            # Never crash on a malformed case: emit a valid persistence
            # forecast so the case stays valid.
            forecast = persistence_forecast(1.0)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
