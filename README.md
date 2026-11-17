# GTA VI launch-day Twitch viewers, hour by hour — round 2

Entry for [make.fun's gta6-twitch-ccu-forecast-01](https://make.fun/c/gta6-twitch-ccu-forecast-01):
forecast how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day, 2026-11-19 (UTC), from everything recorded up to
the end of the day's first hour — as five quantiles (0.05 / 0.25 / 0.5 /
0.75 / 0.95) per hour, hours 2–24.

## What the program does

`main.py` (Python 3.13, standard library only, no network) reads every
`<id>.in` case in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes one
`<id>.out` forecast per case to `$MAKEFUN_OUTPUT_DIR` (else `/output`).

The forecast is a hybrid of two regimes, chosen by the size of the first
hour's print:

1. **Large first hour (regime A).** Per hour, a weighted least-squares fit
   of `log(v_h / v_1)` on `log(p1)` — plus the pre-midnight evening level
   as a second, half-shrunk regressor — over past launches with
   `p1 >= 5,000`. Console-only launches count triple, because their
   rolling-midnight unlock gives them all the same timezone rollout, and in
   the comparables their day curves are far more predictable than PC
   launches' (per-hour residual spread 0.16–0.36 vs 0.46–0.71). The spread
   is the residual spread of the target's own group (console-only when the
   target is one, else the PC group), smoothed over hours and floored at
   0.25 log units; quantiles are lognormal around the fitted median.
2. **Small first hour (regime B).** The round-1 kernel forecast: Gaussian
   kernel-weighted empirical quantiles of the log-ratios and of the absolute
   hourly levels, blended in log space. This is the regime where a launch
   explodes by orders of magnitude once it unlocks, and where the
   log-linear fit of regime A would extrapolate wildly.

The regimes blend in log space with a logistic weight centred at
`p1 = 5,000`, so a case with a large first hour (all four practice cases
and the final) is essentially regime A. Whether the target is console-only
is read from the case: the comparable carrying the target game's own
pre-release history states its platforms, and the practice launches are
known console-only releases. With no usable comparable at all, the program
falls back to persistence with a widening log spread. Output is clamped to
`(0, 1e9]`, non-decreasing, rounded to 2 decimals, and fully deterministic
(no randomness), so two runs are byte-identical.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Local validation

- Format: all 4 practice outputs valid — 23 rows, hours 2–24, monotone
  quantiles in `(0, 1e9]`, files ~1.5 KB (limit 65,536).
- Determinism: two runs produce byte-identical outputs (same sha256).
- Timing: ~0.3 s for all 4 cases together (limit 8 s per case).
- Leave-one-out over the 37 usable comparable launch curves (each curve
  forecast from its hour-1 print and the other curves, scored by log-scale
  pinball loss, through the shipped code): **loss 0.381 overall, 0.244 for
  hour-1 prints >= 5,000, and 0.147 for console-only large curves**,
  versus the round-1 model on the same protocol: 0.385 / 0.269 / 0.219.
  The console-only subset is the relevant regime for this competition:
  GTA VI is console-only, as are all four practice launches.
- Behaviour on the practice cases: medians close to round 1 (evening
  ratios a little higher), spreads much narrower and calibrated
  (q95/q05 ≈ 2.3–3.9× vs round 1's 4–45×; round 1's practice 90%
  coverage was 0.978, i.e. too wide).

## License

MIT — see [LICENSE](LICENSE).
