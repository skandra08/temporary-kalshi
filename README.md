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

### 3. `mmsim`: market making under adverse selection
Hawkes-process order flow with price impact, Avellaneda-Stoikov quoting, and an intensity-aware
extension, evaluated on common random numbers. Validated against analytic fill rates.

### 4. `quantlab`: bias-aware backtesting toolkit
Look-ahead-safe engine, walk-forward validation, Probabilistic/Deflated Sharpe.

## Status
The rain study and the BTC-digitals historical study are complete (above). The options-surface forward test is accumulating data. Next: a breadth screen of ~690 recurring Kalshi series and a forecast-based model for daily-high-temperature ladders.
Forward data accumulates in `data_live/`.

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
