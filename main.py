"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Reads one case file per run (the house's reads of Exophase's Grand Theft
Auto VI PS5 trophy page up to the case's checkpoint) and writes five
quantiles of each trophy's earn rate at the case's target.

Model (docs/methodology.md has the full reasoning, worked examples and
measured numbers):

* The page Exophase shows is frozen between its refreshes, so consecutive
  identical reads are one observation, not many: the history is grouped
  into "epochs", one per distinct frozen page, each dated at the refresh
  that produced it (its first read).
* Per trophy, the earn rate's logit is fitted against ln(1 + hours since
  release). The fit uses only recent epochs: a quadratic in ln-time over
  the last 48 hours when it clearly fits better than a line (its residual
  variance under 70% of the line's), else a line over the last 24 hours
  (48 hours when the last day holds too few epochs). The spread is a
  random-walk term for the horizon plus the fit's prediction variance,
  halved: the house's 0.5 logit-per-root-day random walk is far wider
  than the trend's own error, and the measured loss falls by about half
  when the spread is halved (validated on three simulated final regimes).
* When the history holds no trend at all (a single frozen page, as in the
  practice cases), the forecast is a refresh mixture: Exophase refreshes
  every few hours, and a refresh inside the horizon pulls every rate down
  toward the newly tracked players' lower rates, so the lower quantiles
  sit at the post-refresh level and the upper ones at the current level.
* Everything is deterministic: fixed float arithmetic, no randomness,
  cases read in name order, one output row per trophy.

Run it as the grader does (one case at a time, or a directory of them):

    MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py

Python 3.13, standard library only, no network.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels a case asks for, in the order each row lists them.
DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Random-walk spread of a rate's move, in logit units per square-root day
# (the house baseline's value; the forecast halves the total spread, see
# docs/methodology.md).
SIGMA = 0.5
# The total spread is halved before the quantiles are formed.
SPREAD_SCALE = 0.5
# A quadratic fit must beat a line's residual variance by this factor to be
# used; otherwise the line is the safer extrapolation.
QUAD_VAR_RATIO = 0.7
# Spread floors and caps, in logit units.
SPREAD_MIN = 0.25
SPREAD_MAX = 1.2
# Refresh prior for the no-trend branch: Exophase refreshes every few hours
# (mean interval) and a refresh blends in newly tracked players whose rates
# run about half the current ones.
REFRESH_MEAN_HOURS = 4.0
REFRESH_DILUTION = 0.5


def clamp_pct(x):
    """A rate in percent, clamped to where the logit is defined."""
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(l):
    return 100.0 / (1.0 + math.exp(-l))


def round4(x):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does it.

    A whole number comes back as an int, so json.dumps writes 100 where
    JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def norm_ppf(p):
    """The standard normal's quantile at p (Acklam's approximation)."""
    if p <= 0.0 or p >= 1.0:
        raise ValueError("norm_ppf needs 0 < p < 1")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285984469534e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low = 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > 1.0 - p_low:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def parse_dt(text):
    """An ISO 8601 timestamp (or a bare date) as an aware UTC datetime."""
    text = text.strip()
    if len(text) == 10:
        text += "T00:00:00+00:00"
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def fit_line(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, resid_var, sxx, xbar)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0.0:
        return my, 0.0, 0.0, 0.0, mx
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    a = my - b * mx
    var = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / max(n - 2, 1)
    return a, b, var, sxx, mx


def solve3(mat, vec):
    """Solve a 3x3 system by Gaussian elimination; None when singular."""
    m = [row[:] + [vec[i]] for i, row in enumerate(mat)]
    for col in range(3):
        piv = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        for r in range(3):
            if r != col:
                f = m[r][col] / m[col][col]
                for cc in range(col, 4):
                    m[r][cc] -= f * m[col][cc]
    return [m[i][3] / m[i][i] for i in range(3)]


def fit_quad(xs, ys):
    """Least squares y ~ a + b*x + c*x^2. Returns ([a, b, c], resid_var)."""
    n = len(xs)
    cols = [[1.0] * n, list(xs), [x * x for x in xs]]
    mat = [[sum(cols[i][k] * cols[j][k] for k in range(n)) for j in range(3)]
           for i in range(3)]
    vec = [sum(cols[i][k] * ys[k] for k in range(n)) for i in range(3)]
    beta = solve3(mat, vec)
    if beta is None:
        return None
    var = sum((ys[k] - (beta[0] + beta[1] * xs[k] + beta[2] * xs[k] * xs[k])) ** 2
              for k in range(n)) / max(n - 3, 1)
    return beta, var


def epochs_of(case, release):
    """Group consecutive identical reads into epochs of the frozen page.

    Returns [(t_first_hours, t_last_hours, players_tracked, rates)] where
    the hours count from the release and t_first dates the refresh that
    produced the page (its first read).
    """
    epochs = []
    for read in case["history"]:
        key = (read["players_tracked"], tuple(read["rates"]))
        t = (parse_dt(read["at"]) - release).total_seconds() / 3600.0
        if epochs and epochs[-1][2] == key:
            epochs[-1][1] = t
        else:
            epochs.append([t, t, key])
    return [(e[0], e[1], e[2][0], list(e[2][1])) for e in epochs]


def mixture_quantiles(mu, spread, p_refresh, dilution, levels):
    """Quantiles of a two-part refresh mixture in logit space.

    With probability p_refresh a refresh lands inside the horizon and the
    rate drops to the newly tracked players' level (mu + ln(dilution));
    otherwise it stays around mu. Each part carries the random-walk spread.
    The quantiles come from bisecting the mixture's CDF.
    """
    mu_low = mu + math.log(dilution)

    def ncdf(z):
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    def cdf(x):
        return ((1.0 - p_refresh) * ncdf((x - mu) / spread)
                + p_refresh * ncdf((x - mu_low) / spread))

    lo = mu_low - 6.0 * spread - 1.0
    hi = mu + 6.0 * spread + 1.0
    out = []
    for tau in levels:
        a, b = lo, hi
        for _ in range(60):
            mid = (a + b) / 2.0
            if cdf(mid) < tau:
                a = mid
            else:
                b = mid
        out.append(inv_logit((a + b) / 2.0))
    return out


def forecast_trophy(idx, epochs, t_last, x_target, horizon_days, levels):
    """Five quantiles for one trophy, in the case's level order."""
    zs = [norm_ppf(tau) for tau in levels]
    rw = SIGMA * math.sqrt(horizon_days)
    last_rate = None
    for _, _, _, rates in reversed(epochs):
        if rates[idx] is not None:
            last_rate = rates[idx]
            break
    if last_rate is None:
        last_rate = 0.0
    mu_last = logit(last_rate)

    def points(window_h):
        pts = []
        for t_first, t_last_e, _, rates in epochs:
            if t_last - t_last_e <= window_h and rates[idx] is not None:
                pts.append((math.log1p(t_first), logit(rates[idx])))
        return pts

    def spread_of(var):
        return min(SPREAD_MAX, max(SPREAD_MIN,
                                    SPREAD_SCALE * math.sqrt(rw * rw + var)))

    pts48 = points(48.0)
    pts24 = points(24.0)

    # A quadratic in ln-time over the last 48 hours, when it clearly beats
    # a line there (the trend bends as trophies saturate).
    if len(pts48) >= 5:
        xs = [p[0] for p in pts48]
        ys = [p[1] for p in pts48]
        _, _, var_line, _, _ = fit_line(xs, ys)
        quad = fit_quad(xs, ys)
        if quad is not None:
            beta, var_quad = quad
            if var_quad < QUAD_VAR_RATIO * var_line:
                mu = beta[0] + beta[1] * x_target + beta[2] * x_target * x_target
                s = spread_of(var_quad)
                return [round4(inv_logit(mu + z * s)) for z in zs]

    # A line over the last 24 hours (48 when the last day is too thin).
    for pts in (pts24, pts48):
        if len(pts) >= 3:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            a, b, var, sxx, mx = fit_line(xs, ys)
            mu = a + b * x_target
            pred_var = var * (1.0 + (x_target - mx) ** 2 / sxx) if sxx > 0 else var
            s = spread_of(pred_var)
            return [round4(inv_logit(mu + z * s)) for z in zs]

    # Two points make a line through them, with a wide spread.
    all_pts = points(float("inf"))
    if len(all_pts) == 2:
        (x1, y1), (x2, y2) = all_pts
        if x2 != x1:
            b = (y2 - y1) / (x2 - x1)
            mu = y1 + b * (x_target - x1)
            s = min(SPREAD_MAX, max(SPREAD_MIN, math.sqrt(rw * rw + 0.75 ** 2)))
            return [round4(inv_logit(mu + z * s)) for z in zs]

    # No trend to fit (a single frozen page): expect a refresh.
    horizon_h = horizon_days * 24.0
    p_refresh = 1.0 - math.exp(-horizon_h / REFRESH_MEAN_HOURS)
    qs = mixture_quantiles(mu_last, rw, p_refresh, REFRESH_DILUTION, levels)
    return [round4(q) for q in qs]


def forecast_case(case, name):
    levels = list(case.get("levels") or DEFAULT_LEVELS)
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    release = parse_dt(case["game"]["release_date"])
    target = parse_dt(case["target"])
    x_target = math.log1p((target - release).total_seconds() / 3600.0)
    horizon_days = case["horizon_minutes"] / 1440.0
    epochs = epochs_of(case, release)
    t_last = max(t for _, t, _, _ in epochs)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        qs = forecast_trophy(i, epochs, t_last, x_target, horizon_days, levels)
        # round4 is monotone, but enforce non-decreasing all the same.
        for j in range(1, len(qs)):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
