# Pre-registration: YES-overpricing in Kalshi mention markets

**Frozen:** 2026-10-07 (UTC), by git commit timestamp, *before* any price data for the test samples below
was examined. The hypothesis was generated from the earnings-call results in README section 4 (3,069
word-markets, Jul 2025 - Sep 2026), where buying NO on words priced 15-30c earned +4.1c [+0.7, +7.3]
per contract (one of eight price buckets, so treated as a hint, not a finding).

## Hypothesis
Retail traders over-price YES on cheap-to-mid "will X say WORD" contracts, so a taker who buys NO at the ask
when the YES mid is in (0.15, 0.30] earns a positive net return after the Kalshi fee.

## Rule (fixed, no free parameters)
At the entry time below, if `0.15 < (yes_bid + yes_ask)/2 <= 0.30`, `yes_ask >= yes_bid`, and the spread is
<= 15c, buy one NO at `1 - yes_bid`. Fee = `0.07 * p * (1 - p)` with p the NO price (amortised, series fee
multiplier applied). Payoff 1 if the market resolves NO. Nothing else is traded.

## Two tests (each run once; Bonferroni alpha = 0.025 each)
- **H1b (non-earnings mentions, existing data):** all settled Mentions markets outside the
  `KXEARNINGSMENTION*` series with `close_time` from 2026-08-08 to 2026-10-06 (the window where Kalshi's
  batch candlestick endpoint works). Entry = last quote at or before `open_time + 30 min` (pre-event, since
  these markets open before the event; a YES cannot occur before the event starts).
- **H1a (earnings, prospective):** `KXEARNINGSMENTION*` markets whose call date (from the event ticker) is on or
  after 2026-10-08. Entry = last quote at or before 08:00 UTC on the call date, quote age <= 12h. Interim looks are
  descriptive only; the decision is made once, when at least 300 qualifying contracts have settled.

## Statistic and decision
Mean net P&L per contract, 95% CI by bootstrap clustered on the event (contracts in one event are dependent), and a
one-sided exact binomial test at the event level (H0: win rate <= break-even price + fee). The hypothesis is
supported only if the one-sided p < 0.025 *and* the clustered CI lower bound > 0. Anything else is reported as
not supported, with the point estimate.

## Known limitations declared in advance
Selection: a market that resolves before the entry time has no quote and drops out (negligible for earnings,
possible for non-earnings if an event starts within 30 minutes of the market opening). Taker only; fills at the
quoted ask are assumed; one regime (Aug-Oct 2026 for H1b).
