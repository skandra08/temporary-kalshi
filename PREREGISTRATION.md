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

---

# Addendum H2 (frozen 2026-10-07, before any order-book data is collected): paper-trading a passive NO bidder

**Motivation.** Trade-tape analysis (README, "who loses money in mention markets") shows retail-sized takers who
buy YES lose several cents per contract; the maker side collects. Whether a *new* maker earns that depends on queue
position, which a trade tape alone cannot show, so this test uses live order-book snapshots.

**Ghost order (no real money).** At each snapshot of an open mention market whose best YES ask `a` satisfies
`0.03 <= a <= 0.40`, create a hypothetical resting order buying 1 NO at price `1 - a` (i.e. selling YES at `a`),
joining the back of the queue behind the displayed depth `D` at that level at snapshot time. It fills only when cumulative
taker-YES-buy volume at yes-price `a` since posting exceeds `D` (no credit for queue jumping, and cancellations ahead of
us are assumed not to occur, which is conservative). It is cancelled if the best ask rises above `a`, the market closes, or
60 minutes pass. Maker fee 0 (verified per series; if a series charges maker fees they are deducted).

**Statistic.** Per filled ghost order, P&L = `a - y` (y = 1 if the market resolves YES). Report fill rate, mean P&L per
filled contract with 95% bootstrap CI clustered by event, and the same split by taker trade size. Supported only if the
event-clustered CI lower bound is above 0 with at least 30 events; anything less is reported as not supported, with the
estimate and the number of events. Snapshots are taken every 30 seconds, so ghost orders are overlapping and strongly
dependent: this is why inference is by event, not by order.

**Declared limitations.** Paper trading cannot capture market impact or our own information leakage; cancellations ahead of
us would speed fills (so our fills are conservative) but queue jumpers could take them (so they are optimistic); markets
studied are the most actively traded live mention markets, not a random sample; one regime (Oct 2026 onward).

---

# Amendment A1 to H2 (2026-10-07, before any data from the new universe exists)

The collector originally tracked a fixed list of political mention series whose markets resolve about two weeks out (close
dates Oct 21-23), so a decision (>= 30 settled events) would take months. **The ghost-order rule, statistic and decision criterion
above are unchanged.** Only the *universe* is widened: all Kalshi Mentions series (sports broadcasts, press briefings, TV, politics,
earnings), with priority for markets expected to settle within 3 days, so that settled events accumulate in days. Data already
collected under the old universe stays in the sample. If results differ by market family, they are reported by family, but
the decision uses the pooled sample as pre-specified.
