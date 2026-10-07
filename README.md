# kalshi-edge-research

Quantitative research on **prediction-market pricing**: where Kalshi's prices are efficient,
where they are not, and whether any gap survives fees. Everything is tested on real settled
outcomes with event-clustered inference and walk-forward (never in-sample) evaluation.

## Projects

### 1. `digitaledge`: pricing BTC "above $X" contracts off options and vol models
Kalshi's hourly Bitcoin contracts are digital options. This project prices them from first
principles and scores the result against the market on settled outcomes.
- **Vol models (walk-forward):** HAR-RV with intraday seasonality, EWMA, Deribit DVOL; Gaussian,
  Student-t and empirical tails.
- **Microstructure details:** Kalshi settles on a 60-second average of the index, which shortens
  the effective variance horizon by 2/3 minute; this is modelled explicitly.
- **Options surface:** SVI smile fit to Deribit's live chain; skew-aware digital price
  `N(d2) - vega * d(sigma)/dK`, checked against finite-difference `-dC/dK` and a butterfly
  no-arbitrage test.
- **Inference:** Brier/log-loss with paired cluster-bootstrap by event (strikes in one hour share
  a single outcome, so contracts are not independent).
- **Trading test:** fee-aware backtest (`ceil(0.07 * P * (1-P))` taker fee) with the edge threshold
  chosen on an earlier period and evaluated on a later one; static-arbitrage scan of strike ladders.
- **Forward test:** a collector logs live quotes, the options smile and model prices; settlement
  is joined later, so the options-surface model is evaluated strictly out of sample.

#### Results: 20,354 observations, 468 hourly events (15 Sep - 4 Oct 2026), 95% CIs clustered by event
**Headline (a negative result): Kalshi's hourly BTC ladder is priced better than every volatility
model I built.** Brier(model) minus Brier(market mid), positive = model worse:

![model vs market](results/dbrier.png)

| | 10 min to settle | 30 min to settle |
|---|---|---|
| Gaussian + Deribit DVOL | +0.0084 [0.0059, 0.0110] | +0.0067 [0.0050, 0.0084] |
| Gaussian + EWMA | +0.0056 [0.0036, 0.0078] | +0.0046 [0.0032, 0.0058] |
| Gaussian + HAR-RV (walk-forward) | +0.0040 [0.0024, 0.0058] | +0.0030 [0.0019, 0.0039] |
| **Student-t + HAR-RV** | **+0.0019 [0.0004, 0.0037]** | **+0.0016 [0.0006, 0.0026]** |

- **Slow vol is the wrong tool for an hourly contract.** DVOL (a 30-day implied vol) is the worst
  input, then EWMA, then HAR with intraday seasonality; fat tails (Student-t / empirical) beat
  Gaussian at every horizon.
- **The edge is not tradeable.** Walk-forward fee-aware trading of the best models: +0.1c per
  contract before cent-rounding, **-0.6c after** (CI spans zero); the Gaussian models lose
  1.4-3.2c [CIs below zero].
- **No static arbitrage.** 1 monotonicity violation across 20k quotes, gross 1c, negative net.
- **Where the market wins:** near the money (|distance to strike| < 1 model sigma: gap +0.003,
  CI above zero) and in US trading hours (12-17 UTC: +0.0023 [0.0010, 0.0037]); far from the
  strike the model and market agree (|z| > 1.5: gap ~ 0). The gap also grows as settlement
  approaches (largest at 5 min, ~0 at 45 min), pointing to real-time information.
- **A hypothesis I tested and mostly rejected:** that the market simply has a better spot
  (Kalshi settles on the multi-exchange CF BRTI; I use Coinbase). The ladder-implied center sits
  only ~$4 above Coinbase (+0.03-0.05 sigma). Correcting spot by a walk-forward estimate of that
  basis closes only **5-7%** of the gap, so most of the market's near-the-money edge is
  unexplained (candidates: time-varying index noise, order-flow/momentum information).

Caveats: 20 days, one regime; few observations 60 min out (markets open about an hour before
settlement); the zero-drift martingale assumption ignores BTC's realised drift in the window.
The skew-aware options-surface model cannot be tested on history (no free historical option
surfaces) and is being evaluated forward by the collector.

### 2. Recurring markets: is there an edge in "bet NO on rain, every time"?
Kalshi lists a daily "will it rain in <city>?" market for ~20 US cities (YES if measured
precipitation is strictly above 0 in). `recurring.py` / `rain_study.py` test:
- the naive always-NO rule, by lead time, price bucket and city, net of fees;
- calibration by price (favorite-longshot bias, documented for Kalshi by
  [Whelan 2025](https://www.karlwhelan.com/Papers/Kalshi.pdf));
- a walk-forward forecast model built from *as-of* archived weather forecasts (Open-Meteo
  previous-runs, so no hindsight) against the market price.

#### Results: 1,754 settled markets, 78 days (Jul-Oct 2026), ~20 cities, 95% CIs clustered by day
Rain happened in 23.7% of city-days. Entry prices are Kalshi's quote at the stated time before
close (the climate day starts 24h before close); P&L is one contract bought at the ask, net of the
taker fee `0.07*P*(1-P)`.

![calibration](results/rain_calibration.png)

| Finding | Evidence |
|---|---|
| **The market overprices rain.** | Realised frequency sits below the 45-degree line at almost every price level. At the 24h mark, a YES priced 4-8c rained 0.4-1.3% of the time. |
| **"Always buy NO" is only conditionally true.** | Entered 6-18h before close: +1.4 to +2.5c/contract (CI above zero, e.g. 18h: +2.5c [1.2, 3.7]). Entered 24h before close: +1.4c [0.0, 2.7], borderline. Entered 30-42h before close it **loses** (-4.4c [-6.5, -2.2] at 42h). |
| **The longshot version replicates out of sample.** | Rule "buy NO at day start when YES < 10c", chosen on days 1-39: +1.9c [1.1, 2.4]. Same rule on the held-out days 40-78: **+2.0c [1.3, 2.5]**. The unfiltered "NO on everything" rule at day start is *not* significant in either half (+1.4c [-0.6, 3.3] / +1.3c [-0.5, 3.2]). |
| **It is a pricing quirk, not better forecasting.** | A walk-forward model from *as-of archived* weather forecasts plus city climatology is clearly worse than the market (Brier 0.129 vs 0.086; paired gap +0.042 [0.031, 0.054]) and loses ~1.5c/contract if traded. |

Why a thin edge is not a business: it is ~1-2c on a ~75c contract, and the P&L stream (one NO per
city per day, entered 12h before close) has a daily mean of +$0.41 with a standard deviation of
$0.97 and winning days only 64% of the time; each loss costs ~75c, so it is
"picking up pennies" with a fat left tail. Per-market volume is thousands of contracts, so capacity
is small. With cent-rounded fees on 1-lot orders the edge shrinks by ~0.5c.

**Caveats:** one 78-day window (summer to early autumn) and no winter data; many lead-time and price
cuts were examined (the replication across halves, not any single CI, is the evidence);
fills at the quoted ask are assumed, and depth at the quote is not observed in hourly candles;
quotes older than 3h are dropped. Reproduce with `python scripts/rain_report.py`
(raw numbers in `results/rain_results.json`).

### 3. Breadth screen: do simple rules work across *all* recurring Kalshi markets?
`screen.py` takes 103 daily/weekly/hourly series (weather, economics, commodities, financials,
crypto, entertainment; ~35,700 quotes, ~4,300 events, last ~60 days), buys one contract at the
ask under four simple rules at three points in each market's life (50/80/95% elapsed), and
tests every (series x entry x rule) with each series' own fee multiplier.

**Result: nothing survives.** 736 tests over 88 series; with event-clustered inference,
Benjamini-Hochberg FDR control at 5%, and replication in a held-out half of each series' history,
**0 discoveries**. Pooled by rule (cents per contract, 95% CI clustered by close date):

| Rule | All series | Weather | Economics | Financials |
|---|---|---|---|---|
| Buy NO on everything | **-5.6** [-6.1, -5.1] | -4.3 | -5.4 | -12.9 |
| Buy YES on everything | -3.2 [-3.8, -2.7] | -2.4 | -5.4 | -1.0 (n.s.) |
| Buy YES when priced > 90c | -0.9 [-1.5, -0.4] | -0.1 (n.s.) | +0.1 (n.s.) | n/a |
| Buy NO when YES priced < 10c | -1.6 [-2.0, -1.3] | -1.2 | -0.6 (n.s.) | -6.4 |

- **Taking liquidity at the ask loses roughly the spread plus the fee.** The average quoted spread
  is 7.8c (5.7c in weather); cheap contracts have ~2c spreads, and fees add up to 1.75c at 50c.
- **The rain result does not generalise.** Longshot-NO earned +2c on rain (above) but loses 1.2c
  pooled over 59 weather series, so it is a property of that market, not of Kalshi.
- **Two statistical traps I hit and fixed (both in the commit history):** (1) with only a few
  independent events per series, a cluster bootstrap is degenerate and produced 105 spurious
  "discoveries"; the screen now requires >= 25 events and clusters by event. (2) For
  "favorite at 98c" rules every sampled contract wins, the bootstrap reports zero variance and
  certainty; an exact binomial test at the event level replaced it (45 straight wins at 98c is
  not evidence: 0.98^45 = 40%).
- **What this cannot rule out:** edges of 1-2c at high prices (too few events per series to
  detect), anything on the *maker* side (this tests taking only), and behaviour outside the last
  ~60 days. Raw results: `results/screen_results.csv`, `results/screen_pooled.csv`.

### 4. Mention markets: can ML (incl. low-rank / PCA structure and SEC filing text) beat the crowd?
Kalshi's Mentions category is ~50,000 settled "will X say WORD during EVENT?" markets (earnings calls,
sports broadcasts, press briefings; tens of millions of dollars traded). Data is sparse per company or team, so I tested
whether models that borrow strength can predict which words get said. All evaluations are walk-forward:
features for an event use only events that had *already settled* before it opened (mention markets close
early on YES, so market lifetime leaks the outcome and is never used; prices are taken at a fixed pre-event time).

**Sports announcer markets (539 games, 6 leagues): a null.** A per-word base rate is already strong
(log loss 0.568 vs 0.669 for the series-wide rate), but team-by-word effects and a rank-3 SVD (PCA) smoothing of the
team x word residual matrix add only -0.0014 log loss (95% CI [-0.0042, +0.0012]); gradient boosting overfits (+0.010).
With 30-100 games per series the latent structure is too sparse to learn (`scripts/mentions_sports_experiment.py`).

**Earnings calls (350 calls, 147 companies, 4,708 word-markets): real lift over a base rate, but the market is far sharper.**
Features: each word's count in the company's most recent 10-Q/10-K (SEC EDGAR, filed before the market opened),
a shrunk word prior and company "talkativeness" from earlier settled calls.

| Model (walk-forward) | Log loss | vs per-word base rate (95% CI, clustered by call) |
|---|---|---|
| Per-word base rate | 0.6600 | n/a |
| Word + company prior | 0.6629 | +0.0029 [-0.0008, +0.0066] |
| **+ SEC filing word counts (logistic)** | **0.6424** | **-0.0176 [-0.0257, -0.0097]** |
| Gradient boosting, same features | 0.6469 | -0.0131 [-0.0212, -0.0045] |

The filing text carries real company-specific information that outcome history lacks. **But it is not an edge:**
against Kalshi's own price 8 hours before the call (3,069 word-markets, 230 calls, average spread 3.6c):
- **Accuracy:** market log loss 0.477 vs model 0.638 (gap +0.16, CI [0.14, 0.18]). The market is better in every
  liquidity tercile, including the thinnest (0.37 vs 0.57), so I could not find a pocket where it is less informed.
- **Adds information beyond the market?** Walk-forward stacking of market and model: -0.0032 log loss
  [-0.0091, +0.0028], not significant.
- **Fee-aware trading, threshold chosen on the first half and tested on the second:** -2.2c per contract
  [-4.6, +0.4] for the model; +0.7c [-5.2, +6.2] for the stacked model (n=320, exact p=0.56).
- **A pattern worth testing prospectively, not claiming:** the market looks mildly over-priced on YES for words
  priced 15-30c (realised 15% vs 22% priced; buying NO +4.1c [0.7, 7.3]), but that is one of eight price buckets
  on one season, so it needs a forward test (pre-registered rule, next earnings season).
Scripts: `scripts/earnings_experiment.py`, `scripts/earnings_vs_market.py`; results in `results/earnings_*`.

### 5. `mmsim`: market making under adverse selection
Hawkes-process order flow with price impact, Avellaneda-Stoikov quoting, and an intensity-aware
extension, evaluated on common random numbers. Validated against analytic fill rates.

### 6. `quantlab`: bias-aware backtesting toolkit
Look-ahead-safe engine, walk-forward validation, Probabilistic/Deflated Sharpe.

## Status
The rain study and the BTC-digitals historical study are complete (above). The options-surface forward test is accumulating data. The breadth screen is complete (null). Next: the forecast-based model for daily-high-temperature ladders, a maker-side (quote-providing) test, and the options-surface forward test (needs a local collector).
Forward data accumulates in `data_live/`.

## Run it on your own machine (no cloud needed)
```bash
git clone https://github.com/skandra08/temporary-kalshi && cd temporary-kalshi
bash scripts/setup_local.sh                     # venv + dependencies + tests
source .venv/bin/activate
export EDGAR_UA="kalshi-edge-research you@example.com"   # SEC asks for a contact in the User-Agent
python scripts/earnings_experiment.py           # earnings-call mention markets + SEC filing text
python scripts/mentions_sports_experiment.py    # announcer-mention markets (team / low-rank models)
bash scripts/collect_forever.sh                 # forward data collector (leave running)
```
Notes: every Kalshi and SEC response is cached under `data/`, so runs are resumable. The first earnings
run downloads roughly 400 SEC filings (about 1 GB, 30-60 minutes, limited by SEC and Kalshi rate limits).

## Run it
```bash
git clone https://github.com/skandra08/kalshi-edge-research && cd kalshi-edge-research
pip install -e ".[dev]"
pytest -q                                   # 33 tests
python -m digitaledge study --start 2026-09-15 --end 2026-10-05
bash scripts/collect_forever.sh             # local forward-data collector
python -m digitaledge.live                  # one live snapshot vs the options surface
```
Public API responses are cached under `data/`, so reruns are fast and resumable (Kalshi
rate-limits, so first pulls are slow).

## Layout
```
digitaledge/  data sources, vol models, digital pricing, SVI, scoring, backtest, rain study, collector
mmsim/        Hawkes market-making simulator
quantlab/     backtesting toolkit
tests/        33 tests, all offline (synthetic data with known ground truth)
scripts/      local collector loop
data_live/    forward snapshots (gzip CSV)
```
