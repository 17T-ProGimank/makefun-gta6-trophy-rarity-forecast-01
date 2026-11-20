# Methodology — GTA VI trophy rarity forecast

## What is being forecast

Each trophy's earn rate on Exophase's Grand Theft Auto VI PS5 page at the
case's target: the percent of the players Exophase tracks who have earned
that trophy. The score is the mean pinball loss on the logit scale over
every trophy and quantile; lower is better. A prize needs a loss strictly
below the best house entry (persistence baseline, trend baseline, house
AI-model forecast).

## What the data looks like

Exophase does not update its page continuously: it refreshes a game's
counts every few hours, and between refreshes the page is frozen. The
house's reads therefore repeat the same page many times. Consecutive
identical reads are **one observation, not many**, so the program groups
the history into *epochs* — one per distinct frozen page, dated at the
refresh that produced it (its first read). On launch night the page was
frozen at 638 players tracked from 23:48Z through the 04:00Z read, then
refreshed by 04:30Z to 3,868 players with every rate but the 100% one
lower: the newly tracked players held fewer trophies. That refresh is
the whole story of the practice cases: their histories are a single
frozen page and their answers are the refreshed page.

## The model

Per trophy, the earn rate's logit is fitted against `ln(1 + hours since
release)`:

1. **Trend (the final's regime).** A quadratic in ln-time over the last
   48 hours of epochs is used when it clearly beats a line there (its
   residual variance under 70% of the line's) — trophies saturate as the
   reachable players finish them, so the trend bends. Otherwise a line
   over the last 24 hours of epochs (48 hours when the last day holds
   too few epochs). The spread is a random-walk term for the horizon
   plus the fit's prediction variance, **halved**: the house's 0.5
   logit-per-root-day random walk is far wider than the trend's own
   error.
2. **No trend (the practice regime).** When the history holds a single
   frozen page there is nothing to fit, so the forecast is a **refresh
   mixture**: with probability `p = 1 - exp(-horizon / 4h)` a refresh
   lands inside the horizon and the rate drops to the newly tracked
   players' level (about half the current rate); otherwise it stays
   where it is. Each part carries the random-walk spread, and the five
   quantiles come from bisecting the mixture's CDF. So the lower
   quantiles sit near the post-refresh level and the upper ones near the
   current level.
3. **Fallbacks.** Two points make a line through them, with the spread
   halved like the trend branches (the 0.75 logit allowance for the
   line's own error goes through the same spread rule); a trophy whose
   rate is missing everywhere falls back to the mixture centred on its
   last non-null rate. Rates are clamped to 0.01–99.99 before the logit,
   as the scorer does.

Everything is deterministic: fixed float arithmetic, no randomness,
cases read in name order.

## How it was measured

The kit ships no local grader and the practice answers are sealed on
the server, so the practice answers were **reconstructed** from the
three figures the rules publish about the house baseline's practice
grade: the 04:30Z read's 4,276 trophies earned over 3,868 players
(the answer rates sum to 110.55), a typical miss of about 0.68
percentage points, and a baseline loss of about 1.51 with 90%
coverage 0. A power-law drop `y = 0.2304 * r^2` (100% trophy kept at
100%) reproduces all three at once: baseline loss **1.5121**, typical
miss **0.6771**, coverage **0.0**. A proportional drop (`y = 0.2208 r`)
matches the sum and the miss but gives a baseline loss of only 1.40,
so it is kept as a sensitivity check, not the primary proxy.

Practice loss (both cases, mean over all trophies and quantiles):

| forecast | proxy B (power-law) | proxy A (proportional) |
|---|---|---|
| house persistence baseline | 1.5121 | 1.4008 |
| this entry | **1.1178** | **1.0062** |

The entry beats the best house baseline on practice with margin under
both reconstructions.

The final is a different regime (a 4.5-day horizon over a real trend),
so the trend branch was validated on simulated finals: synthetic
Exophase worlds anchored to the real launch-night rates (638 players,
rates 0.63–2.04% plus one at 100%), with logistic players-tracked
growth, per-trophy progression and dilution, and random refresh
cadences. Three worlds (smooth trends; saturating trends; saturating
trends plus a post-deadline slope shift and level jump), 30 draws each:

| forecast | smooth | saturating | shock |
|---|---|---|---|
| house persistence baseline | 0.3050 | 0.3189 | 0.3072 |
| house trend baseline (last 24 h) | 0.2570 | 0.3004 | 0.3072 |
| this entry | **0.1105** | **0.1424** | **0.1941** |

The entry beats the house trend baseline in 30/30 draws in every
world. The two choices that do the work are the epoch grouping
(one point per frozen page instead of one per read) and the halved
spread; the quadratic branch adds the rest on saturating trends.

## Worked example

Practice case p00, trophy 4 (last rate 1.10%, reconstructed answer
0.2788%). The mixture uses `p = 1 - exp(-1.5/4) = 0.3127`,
spread `0.5 * sqrt(90/1440) = 0.125`, `logit(1.10) = -4.4988` and
the post-refresh centre `logit(1.10) + ln(0.5) = -5.1919`
(0.553%). The entry's row is `[0.4887, 0.6139, 1.0207, 1.1484,
1.3166]`; the baseline's is `[0.8974, 1.012, 1.1, 1.1955, 1.3477]`.
Pinball items at the five levels (2-decimal answer, logit scale):

| level | entry | baseline |
|---|---|---|
| 0.05 | 1.0704 | 2.2329 |
| 0.25 | 1.1891 | 1.9449 |
| 0.50 | 1.3052 | 1.3808 |
| 0.75 | 0.7122 | 0.7325 |
| 0.95 | 0.1563 | 0.1586 |
| **mean** | **0.8866** | **1.2900** |

The win comes from the low quantiles: they sit near the refreshed
level instead of the frozen one.

## Verification (local, 20 Nov 2026)

- Both practice cases run: outputs valid (55 trophies, five
  quantiles each, 0–100, non-decreasing, 2,808 bytes, under the
  65,536-byte cap), two runs byte-identical, 0.13 s for both cases
  (limit 8 s per case).
- Synthetic final cases with real ISO timestamps run end to end
  (losses 0.11–0.16 across worlds), including edge cases: a
  two-epoch history, null rates in some reads, and a trophy added
  late in the history (nulls before it appears).
