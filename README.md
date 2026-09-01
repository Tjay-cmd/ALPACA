# ALPACA — Volatility Risk Premium Options Agent

Hackathon submission: an autonomous options agent on Alpaca paper trading. An LLM proposes defined-risk verticals inside hard-coded constraints; an independent risk gate re-prices every trade from live quotes and can reject it entirely.

---

## 1. Project overview

This agent trades listed equity and ETF options on Alpaca using a **volatility risk premium** signal: implied volatility (IV) versus realized volatility (RV). When IV looks rich versus RV it sells premium (credit spreads); when IV looks cheap it buys premium (debit spreads). A language model chooses the specific spread, strikes, and expiration **only from directions the signal allows**. `risk.py` then independently re-checks live bid/ask, size, and fill quality. If that check fails, nothing is sent to Alpaca.

---

## 2. Strategy / the core thesis

Options markets usually price **more** volatility than the underlying later realizes. That gap is the **volatility risk premium** (sometimes discussed as the variance risk premium): implied vol tends to sit above subsequent realized vol because option buyers pay for insurance and sellers demand compensation for tail risk. This is a documented market regularity, not a rule we invented — it shows up in index and single-name options research and in practitioner vol-selling literature.

The agent turns that into a two-sided rule:

| IV vs RV | Interpretation | Allowed trade |
|---|---|---|
| IV − RV **> +0.02** | Premium looks **rich** | Credit spread (`bull_put` or `bear_call`) or stand aside |
| IV − RV **< −0.02** | Premium looks **cheap** | Debit spread (`bull_call` or `bear_put`) or stand aside |
| Inside ±0.02 | Signal too small | `no_trade` only |

IV is an ATM-weighted average of live chain implied vols in a **20–45 DTE** window. RV is annualized 20-day realized vol from daily log returns: `std(log returns) × √252`. The live hurdle of **0.02** (two volatility points) is the same gate used in `allowed_actions()`.

The LLM does **not** get to invent a new thesis. It only picks structure and strikes inside that box.

---

## 3. Architecture

The live path is a single pass per symbol. No stage trusts the previous stage’s dollar amounts.

```
Alpaca market data          Featherless LLM
 (bars + option chain)       (JSON decision)
         |                         |
         v                         v
   data.py ----> vol_signal.py ----> decision.py
   prices, IV,            IV/RV spread,     action + strikes
   greeks                 allowed_actions()  (direction-capped)
                                   |
                                   v
                                risk.py
                     live bid/ask, size, fill quality
                     LLM contracts / max_loss ignored
                                   |
                          pass / reject
                                   |
                                   v
                              execute.py
                     Alpaca MLEG limit order
                     trade_log.csv + scheduler.log
                                   |
                                   v
                           scheduler.py
                     market clock, 15-min loop
                     systemd on DigitalOcean
```

| Stage | File | What it does |
|---|---|---|
| Data | `data.py` | Daily bars and the current option chain from Alpaca (stock feeds IEX then SIP; options INDICATIVE then OPRA). Parses OCC symbols for strike / expiry / put-call. |
| Signal | `vol_signal.py` | 20-day RV and the IV/RV spread. This is the strategy brain, not the LLM. |
| Decision | `decision.py` | Featherless (`Qwen/Qwen2.5-7B-Instruct`, with Llama-3.1-8B and Hermes-3-8B fallbacks). Prompted only with `allowed_actions()`. After parse, any illegal action is forced to `no_trade`. |
| Risk | `risk.py` | Independent verification. Re-looks-up both legs, uses **conservative** fills (sell at bid, buy at ask), recalculates credit/debit and max loss, clamps size to **2% of equity**, caps **3** open option positions. |
| Execute | `execute.py` | Two-leg `LimitOrderRequest` with `OrderClass.MLEG` on the paper account. Logs every attempt. Prints `ABOUT TO SUBMIT` before a live send. |
| Schedule | `scheduler.py` | Alpaca `get_clock()` — skips work when the market is closed. 15-minute cycle, max 20 symbol-runs per day. File + console logging. |

**Tech stack:** Python 3, [alpaca-py](https://github.com/alpacahq/alpaca-py) (Trading API + market data), [Featherless](https://featherless.ai) (OpenAI-compatible LLM inference), Streamlit dashboard (`dashboard.py`) for the trade log, DigitalOcean droplet + `systemd` (`trading-agent.service`) for unattended 24/7 operation.

**Alpaca surfaces used**

- Paper `TradingClient`: account equity, positions, clock/calendar, multi-leg option orders
- Stock historical bars for RV and the proxy backtest
- Option chain snapshots with implied vol and greeks for the live signal
- Limit-only MLEG verticals (no market orders)

---

## 4. The risk gate — why it matters

The LLM **proposes**. It never has final authority.

`risk.py` does not reuse the model’s `contracts` or `max_loss_dollars`. It looks the short and long strikes back up on the **current** chain, requires strictly positive bid/ask on both legs, and prices the spread at the worst reasonable fill. For a credit spread, conservative net credit is `short_bid − long_ask`. If that number is **≤ 0**, the trade is rejected — even if the model was confident and the narrative sounded right. Size is then `floor((equity × 2%) / max_loss_per_contract)`. Oversized proposals are clamped, not trusted.

`execute.py` will not submit unless `risk_check_passed` is true. Failed gates still get a `trade_log.csv` row so rejections are visible.

**The LLM optimizes within constraints. It does not have unilateral control.** That is the safety argument, and it is not theoretical — two real incidents during development showed why.

### Incident A — AAPL: broken credit, gate said no

On 2026-08-30, a multi-symbol dry run (`test_multi_symbol.py`) produced:

```
AAPL  | RV: 0.188 | IV: 0.314 | Spread: +0.126 | Decision: credit_spread | Risk: FAIL
```

The model wanted a **bear call credit** while IV was rich versus RV (a legal *direction*). The structure it picked was not a real credit. The logged proposal was short 320 / long 315 — inverted for a call credit — and live bid/ask implied a **non-positive conservative net credit** (crossed or too-wide quotes). The gate refused the fill and wrote `action=no_trade`, `risk_check_passed=False` to `trade_log.csv`. No order was placed.

A later pass on AAPL (320 / 325) did pass sizing. The point is the first one: a fluent, on-thesis credit proposal still died when the quotes could not support a credit. The model does not get to “talk past” the book.

### Incident B — NVDA: wrong direction, then a hard constraint

The same dry run, **before** the direction lock:

```
NVDA  | RV: 0.465 | IV: 0.438 | Spread: -0.027 | Decision: credit_spread | Risk: PASS
```

IV was **below** RV. The strategy says that is cheap premium: debit or stand aside, never a credit. The LLM still chose a `bear_call` credit, and the *quote* check passed — so a direction bug would have been paper-traded if we had been live.

That is a different failure mode from AAPL: the risk gate checks **money and fills**, not thesis. We fixed it in `decision.py` with `allowed_actions()`:

- spread > +0.02 → `["credit_spread", "no_trade"]`
- spread < −0.02 → `["debit_spread", "no_trade"]`
- otherwise → `["no_trade"]`

The list is injected into the prompt **and** enforced after parse. Any illegal action is overridden to `no_trade`. Re-running the same pipeline, NVDA was restricted to debit / no trade; it chose a debit and the risk gate then failed it on quotes (`Conservative net debit is -1.79`). Wrong-way credits on a negative spread stopped.

Two layers, two incidents: quotes can veto a “good story,” and code — not the model — owns which side of the vol premium is even discussable.

---

## 5. Validation / backtesting

Alpaca’s retail data path does not give a deep historical options tape we can replay into real vertical P&L. We therefore **do not claim** a historical options backtest.

What we did instead is a **proxy** test of the core premise: when 20-day realized vol is rich or cheap versus a 90-day baseline, does realized vol over the next 20 sessions tend to mean-revert? That is the economic story behind selling rich vol and buying cheap vol. It is **not** fills, IV, theta, or spread P&L.

`backtest.py` ran that proxy on the original six names (SPY, QQQ, AAPL, TSLA, NVDA, IWM). Alpaca returned **539 daily bars, 2024-07-08 to 2026-08-28 (~2.14 years)** per name. At the live **0.02** gap, non-overlapping signals aggregated to:

| Side | Signals | Win rate | Avg forward vol change (wins / losses) | Worst adverse |
|---|---|---|---|---|
| Sell premium (rv20 > rv90 + 0.02) | 53 | 67.9% | −0.132 / +0.162 | +0.470 |
| Buy premium (rv20 < rv90 − 0.02) | 69 | 58.0% | +0.076 / −0.051 | −0.197 |

Per-symbol counts were small (roughly 7–12 each). The script labels those as directional only, not statistically robust.

`backtest_threshold_sweep.py` extends the same signal logic to **18 liquid, optionable names** (the original six plus DIA, XLF, XLE, XLV, GLD, TLT, MSFT, GOOGL, AMZN, META, JPM, AMD) and sweeps thresholds `[0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]`. Each symbol’s history is split **in half chronologically** (first half train, second half test). A threshold is treated as real evidence of edge only if win rate holds up on **both** halves with a usable sample. Thresholds that look great in one window and mediocre in the other are printed as **overfitting warnings**. More extreme hurdles shrink n and were treated as likely noise; the **live gate stays at 0.02**, which is also the original proxy hurdle that showed a supportive sell-side rate without collapsing the sample.

**Known limitation (please read this as part of the submission, not a footnote):** this validates that the vol mean-reversion *premise* is not obviously wrong in this window. It is **not** proof of profitable options trading. Paper and live spread P&L can differ by a lot because of IV, bid/ask, slippage, theta, and strike selection. Testing many thresholds on the same prices can discover noise; the split-half check is a partial safeguard, not a guarantee.

---

## 6. Live trading setup

A **dedicated Alpaca paper account** with the default **$100,000** starting balance was created for the official competition measurement window, as required by the hackathon rules. It is **not** the account used for local development, dry runs, and pipeline debugging. Competition P&L should be read from that measurement account only.

The agent runs unattended on a DigitalOcean droplet:

- Working directory `/root/ALPACA`, venv Python, `scheduler.py`
- `systemd` unit `trading-agent.service`: `Restart=on-failure`, enabled on boot
- Alpaca clock: **no data, LLM, or orders when the market is closed**
- Open-hours loop: every **15 minutes**, day-one symbols **SPY / QQQ / IWM** (most liquid names; fuller list commented in `scheduler.py`)
- Cap of **20** symbol pipeline runs per calendar day
- Rotating `scheduler.log` (10MB × 5) plus the journal

No one has to sit on a terminal for it to trade. See `DEPLOYMENT.md` for install and log commands.

Hard live limits (unchanged by the LLM): **2% of equity** max loss per new spread, **3** open option positions, limit orders only, paper endpoint only.

---

## 7. Disclosure — pre-event / setup

**All setup and infrastructure work happened during the official hackathon week, not before it.**

This was not a finished production agent we brought to the event and “turned on.” During the week we:

- Created the Alpaca paper credentials and wired `alpaca-py` (account, clock, bars, option chain, MLEG orders)
- Stood up Featherless inference and the decision JSON contract
- Wrote the pipeline (`data` → `signal` → `decision` → `risk` → `execute`), the scheduler, logging, and the Streamlit log view
- Built the proxy backtest and the split-half threshold sweep
- Deployed the droplet, venv, `.env`, and systemd unit
- Opened a **separate** paper account for the official measurement window so development fills would not pollute scored P&L

We are disclosing this so judges can treat the project as **built in-week**, including accounts, cloud, and APIs — not as a pre-existing shop strategy dropped into Alpaca for the leaderboard.

---

## Local setup

1. Create and activate a virtual environment:

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

   macOS / Linux:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and set paper `APCA_API_KEY_ID`, `APCA_API_SECRET_KEY`, and `FEATHERLESS_API_KEY`.

4. Smoke-test Alpaca:

   ```bash
   python test_connection.py
   ```

5. One symbol (respects `DRY_RUN` in `main.py`):

   ```bash
   python main.py
   ```

6. Scheduler (headless; same `DRY_RUN` flag):

   ```bash
   python scheduler.py
   ```

Cloud install: `DEPLOYMENT.md`.

---

## Repository map

| Path | Role |
|---|---|
| `data.py` | Alpaca prices and option chain |
| `vol_signal.py` | RV and IV/RV spread |
| `decision.py` | Featherless LLM + `allowed_actions()` |
| `risk.py` | Independent quote/size gate |
| `execute.py` | MLEG submit + `trade_log.csv` |
| `main.py` | One full pass |
| `scheduler.py` | Market-hours loop |
| `dashboard.py` | Streamlit trade log |
| `backtest.py` | Proxy vol mean-reversion test |
| `backtest_threshold_sweep.py` | Multi-symbol threshold + split-half |
| `trading-agent.service` | systemd unit |
| `DEPLOYMENT.md` | Droplet install |

`.env` is not in git.
