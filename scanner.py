import os
import asyncio
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from telegram import Update
from telegram.request import HTTPXRequest
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
OWNER_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()
INTERVAL_SECONDS = max(
    60, int(os.getenv("SCAN_INTERVAL_SECONDS", "300"))
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("advanced-scanner")

# Yahoo Finance reference feeds; not guaranteed to match Exness quotes.
SYMBOLS = {
    "XAUUSD": "GC=F",
    "XAGUSD": "SI=F",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
}

TIMEFRAMES = {
    "1m": ("1m", "1d"),
    "5m": ("5m", "5d"),
    "15m": ("15m", "1mo"),
    "30m": ("30m", "1mo"),
    "1h": ("60m", "3mo"),
    "4h": ("60m", "3mo"),
}

WATCHING = set()
LAST_ALERT = {}
LAST_CANDLE = {}


def allowed(update: Update) -> bool:
    if not OWNER_CHAT_ID:
        return False
    chat = update.effective_chat
    return bool(chat and str(chat.id) == OWNER_CHAT_ID)


async def guard(update: Update) -> bool:
    chat = update.effective_chat
    user = update.effective_user
    msg = update.effective_message
    log.info(
        "COMMAND CHECK: chat_id=%s user_id=%s text=%r",
        chat.id if chat else None,
        user.id if user else None,
        (msg.text[:100] if msg and msg.text else None),
    )
    if not allowed(update):
        if update.effective_message:
            await update.effective_message.reply_text(
                "Access denied. Bot owner only."
            )
        return False
    return True


def download_biquote_candles(interval):
    import json
    from urllib.request import Request, urlopen

    allowed = {"1m", "5m", "15m", "30m", "1h"}
    if interval not in allowed:
        raise ValueError(f"Unsupported timeframe: {interval}")

    url = f"https://biquote.io/api/XAUUSD/ohlc?interval={interval}&limit=500"
    req = Request(url, headers={"User-Agent": "AdvancedTelegramScanner/1.0"})

    # Retry temporary network failures without changing the provider URL or data.
    payload = None
    last_error = None

    for attempt in range(3):
        try:
            with urlopen(req, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except (TimeoutError, OSError) as exc:
            last_error = exc
            if attempt == 2:
                raise
            import time
            time.sleep(2 * (attempt + 1))

    if payload is None:
        raise RuntimeError(f"BiQuote request failed: {last_error}")

    if payload.get("symbol", "").upper() != "XAUUSD":
        raise ValueError("Unexpected symbol returned by data provider")

    bars = payload.get("bars")
    if not isinstance(bars, list) or not bars:
        raise ValueError("No candles returned")

    df = pd.DataFrame(bars)
    required = {"openTime", "open", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing candle fields: {required - set(df.columns)}")

    if "isOpen" in df.columns:
        open_flags = df["isOpen"].map(
            lambda x: x if isinstance(x, bool)
            else str(x).strip().lower() == "true"
        )
        df = df[~open_flags].copy()

    df["Timestamp"] = pd.to_datetime(df["openTime"], utc=True, errors="coerce")
    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume"
    })

    if "Volume" not in df.columns:
        df["Volume"] = 0

    for col in ("Open", "High", "Low", "Close", "Volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Timestamp", "Open", "High", "Low", "Close"])
    df = df.set_index("Timestamp").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    if len(df) < 50:
        raise ValueError(f"Only {len(df)} completed candles received")

    age = (pd.Timestamp.now(tz="UTC") - df.index[-1]).total_seconds()
    max_age = {"1m": 600, "5m": 1500, "15m": 3600,
               "30m": 7200, "1h": 4500}[interval]
    if age < -60 or age > max_age:
        raise ValueError(f"Latest completed candle is stale or future-dated: {age:.0f}s")

    return df[["Open", "High", "Low", "Close", "Volume"]]


def download_candles(symbol, interval):
    symbol = symbol.upper()

    if symbol == "XAUUSD" and interval in ("1m", "5m", "15m", "30m", "1h"):
        return download_biquote_candles(interval)

    if symbol == "XAUUSD" and interval == "4h":
        # Preserve the existing 4h aggregation logic.
        hourly = download_biquote_candles("1h")
        agg = hourly.resample("4h", label="left", closed="left").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }).dropna(subset=["Open", "High", "Low", "Close"])
        now = pd.Timestamp.now(tz="UTC")
        if len(agg) and (agg.index[-1] + pd.Timedelta(hours=4)) > now:
            agg = agg.iloc[:-1]
        if len(agg) < 40:
            raise ValueError(
                f"Not enough 4h candles ({len(agg)}); provider has limited history."
            )
        return agg

    if interval not in TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe: {interval}")
    if symbol not in SYMBOLS:
        raise ValueError(f"Unsupported symbol: {symbol}")

    yf_interval, period = TIMEFRAMES[interval]
    df = yf.download(
        SYMBOLS[symbol],
        period=period,
        interval=yf_interval,
        progress=False,
        auto_adjust=False,
        threads=False,
    )

    if df is None or df.empty:
        raise ValueError(f"No Yahoo Finance candles for {symbol} {interval}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing Yahoo candle columns: {missing}")

    if "Volume" not in df.columns:
        df["Volume"] = 0

    df = df.dropna(subset=required).copy()
    minimum_candles = 50 if interval in ("1m", "5m", "15m", "30m", "60m") else 210
    if len(df) < minimum_candles:
        raise ValueError(
            f"Not enough candles ({len(df)}) for {interval}; "
            f"need at least {minimum_candles}."
        )

    now = pd.Timestamp.now(tz="UTC")
    idx = pd.to_datetime(df.index, utc=True)
    latest_age = (now - idx[-1]).total_seconds()
    max_age = {
        "1m": 10 * 60,
        "5m": 25 * 60,
        "15m": 60 * 60,
        "30m": 2 * 60 * 60,
        "1h": 4 * 60 * 60,
    }.get(interval, 4 * 60 * 60)

    if latest_age < -60 or latest_age > max_age:
        raise ValueError(
            f"Stale {interval} feed: newest candle is "
            f"{latest_age / 60:.0f} minutes old."
        )

    return df[["Open", "High", "Low", "Close", "Volume"]]
def add_indicators(df):
    df = df.copy()
    c = df["Close"]
    h = df["High"]
    l = df["Low"]

    df["EMA20"] = c.ewm(span=20, adjust=False).mean()
    df["EMA50"] = c.ewm(span=50, adjust=False).mean()
    df["EMA200"] = c.ewm(span=200, adjust=False).mean()

    delta = c.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()
    avg_loss = loss.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(
        span=9, adjust=False
    ).mean()

    previous_close = c.shift(1)
    true_range = pd.concat(
        [
            h - l,
            (h - previous_close).abs(),
            (l - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    df["ATR"] = true_range.ewm(
        alpha=1 / 14, min_periods=14, adjust=False
    ).mean()

    df["GREEN"] = (c > df["Open"]).astype(int)
    return df.dropna()


def analyze(symbol, timeframe):
    df = add_indicators(download_candles(symbol, timeframe))

    # 4h has limited provider history; other timeframes retain the 50-row rule.
    minimum_rows = 25 if timeframe == "4h" else 50
    if len(df) < minimum_rows:
        raise ValueError(
            f"Not enough completed candles ({len(df)}); "
            f"need at least {minimum_rows} for {timeframe}."
        )

    # Last completed indicator row; retain enough history for previous-candle checks.

    row = df.iloc[-2]
    previous = df.iloc[-3]
    last_candle_time = str(df.index[-2])

    price = float(row["Close"])
    atr = float(row["ATR"])
    rsi = float(row["RSI"])

    if not np.isfinite(atr) or atr <= 0:
        raise ValueError("Invalid ATR; cannot calculate risk levels.")

    score = 0
    reasons = []

    if row["EMA20"] > row["EMA50"]:
        score += 1
        reasons.append("EMA20 above EMA50")
    else:
        score -= 1
        reasons.append("EMA20 below EMA50")

    if price > row["EMA200"]:
        score += 2
        reasons.append("Price above EMA200")
    else:
        score -= 2
        reasons.append("Price below EMA200")

    if row["MACD"] > row["MACD_SIGNAL"]:
        score += 1
        reasons.append("MACD bullish")
    else:
        score -= 1
        reasons.append("MACD bearish")

    if 50 < rsi < 70:
        score += 1
        reasons.append("RSI supports bullish momentum")
    elif 30 < rsi < 50:
        score -= 1
        reasons.append("RSI supports bearish momentum")
    elif rsi >= 70:
        reasons.append("RSI overbought: caution")
    else:
        reasons.append("RSI oversold: caution")

    recent = df.iloc[-22:-2]
    green_pct = float(recent["GREEN"].mean() * 100)
    if green_pct > 60:
        score += 1
        reasons.append("More green candles recently")
    elif green_pct < 40:
        score -= 1
        reasons.append("More red candles recently")

    # Estimate pressure from completed candle OHLC, not order-book data.
    pressure_window = df.iloc[-22:-2].copy()
    candle_range = (
        pressure_window["High"] - pressure_window["Low"]
    ).replace(0, np.nan)

    close_location = (
        2 * pressure_window["Close"]
        - pressure_window["High"]
        - pressure_window["Low"]
    ) / candle_range

    close_location = close_location.replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if len(close_location) < 10:
        raise ValueError("Not enough valid candles to estimate pressure.")

    mean_pressure = float(close_location.mean())
    pressure_index = round(
        max(0.0, min(100.0, 50.0 + 50.0 * mean_pressure)), 1
    )
    bullish_candles = int((close_location > 0).sum())
    bearish_candles = int((close_location < 0).sum())

    if pressure_index >= 60:
        pressure_label = "BUYING PRESSURE ESTIMATE"
    elif pressure_index <= 40:
        pressure_label = "SELLING PRESSURE ESTIMATE"
    else:
        pressure_label = "MIXED / NEUTRAL PRESSURE"

    # Higher timeframe trend confirmation for intraday signals.
    higher_tf = {
        "1m": "5m",
        "5m": "15m",
        "15m": "1h",
        "30m": "1h",
        "1h": "4h",
        "4h": "1h",
    }[timeframe]
    htf_confirm = "unavailable"

    try:
        higher = add_indicators(
            download_candles(symbol, higher_tf)
        )
        hr = higher.iloc[-2]
        if hr["EMA50"] > hr["EMA200"]:
            htf_confirm = "bullish"
        elif hr["EMA50"] < hr["EMA200"]:
            htf_confirm = "bearish"
        else:
            htf_confirm = "neutral"
    except Exception as exc:
        log.warning("Higher timeframe unavailable: %s", exc)

    if htf_confirm == "bullish":
        score += 1
        reasons.append(f"{higher_tf} trend bullish")
    elif htf_confirm == "bearish":
        score -= 1
        reasons.append(f"{higher_tf} trend bearish")

    # Strict confirmation: weak or conflicting evidence means WAIT.
    macd_bullish = row["MACD"] > row["MACD_SIGNAL"]
    macd_bearish = row["MACD"] < row["MACD_SIGNAL"]
    ema_bullish = row["EMA20"] > row["EMA50"]
    ema_bearish = row["EMA20"] < row["EMA50"]
    above_ema200 = price > row["EMA200"]
    below_ema200 = price < row["EMA200"]

    if (
        score >= 4
        and rsi < 68
        and macd_bullish
        and ema_bullish
        and above_ema200
        and htf_confirm == "bullish"
    ):
        action = "BUY"
    elif (
        score <= -4
        and rsi > 32
        and macd_bearish
        and ema_bearish
        and below_ema200
        and htf_confirm == "bearish"
    ):
        action = "SELL"
    else:
        # Secondary setup alerts: NOT confirmed trade entries.
        # Keep strict BUY/SELL conditions above unchanged.
        if (
            score >= 2
            and ema_bullish
            and macd_bullish
            and above_ema200
            and rsi < 68
            and htf_confirm == "bullish"
        ):
            action = "BUY SETUP"
            reasons.append(
                "Bullish setup forming; strict entry confirmation is not met."
            )
        elif (
            score <= -2
            and ema_bearish
            and macd_bearish
            and below_ema200
            and rsi > 32
            and htf_confirm == "bearish"
        ):
            action = "SELL SETUP"
            reasons.append(
                "Bearish setup forming; strict entry confirmation is not met."
            )
        else:
            action = "WAIT"

        if action == "WAIT":
            failed_checks = []
        if htf_confirm not in ("bullish", "bearish"):
            failed_checks.append(f"{higher_tf} trend is {htf_confirm}")
        elif score >= 2 and htf_confirm != "bullish":
            failed_checks.append(f"{higher_tf} trend conflicts with bullish bias")
        elif score <= -2 and htf_confirm != "bearish":
            failed_checks.append(f"{higher_tf} trend conflicts with bearish bias")
        if not (macd_bullish or macd_bearish):
            failed_checks.append("MACD has no clear direction")
        if not (ema_bullish or ema_bearish):
            failed_checks.append("EMA20 equals EMA50")
        if score >= 2 and rsi >= 68:
            failed_checks.append("RSI too high for BUY confirmation")
        if score <= -2 and rsi <= 32:
            failed_checks.append("RSI too low for SELL confirmation")
        if not failed_checks:
            failed_checks.append("full entry confirmation rules not met")
        reasons.append("WAIT details: " + "; ".join(failed_checks))

    # ATR-based planning levels; verify prices with the broker.
    if action in ("BUY", "BUY SETUP"):
        entry_low = price
        entry_high = price + 0.15 * atr
        entry = price
        sl = entry - 1.5 * atr
        tp1 = entry + 2.25 * atr
        tp2 = entry + 3.0 * atr
        tp3 = entry + 4.5 * atr
    elif action in ("SELL", "SELL SETUP"):
        entry_low = price - 0.15 * atr
        entry_high = price
        entry = price
        sl = entry + 1.5 * atr
        tp1 = entry - 2.25 * atr
        tp2 = entry - 3.0 * atr
        tp3 = entry - 4.5 * atr
    else:
        entry_low = entry_high = None
        entry = sl = tp1 = tp2 = tp3 = None

    if action in ("BUY SETUP", "SELL SETUP"):
        entry_low = entry_high = None
        entry = sl = tp1 = tp2 = tp3 = None

    # Directional forecast scenarios; never guaranteed targets.
    if score >= 2:
        bias, forecast_sign = "BULLISH", 1
    elif score <= -2:
        bias, forecast_sign = "BEARISH", -1
    else:
        bias, forecast_sign = "NEUTRAL / UNCLEAR", 0

    forecast_1 = price + forecast_sign * atr
    forecast_2 = price + forecast_sign * 2 * atr
    forecast_3 = price + forecast_sign * 3 * atr

    pip_size = {
        "EURUSD": 0.0001,
        "GBPUSD": 0.0001,
        "USDJPY": 0.01,
    }.get(symbol)

    def pip_move(target):
        if pip_size is None or forecast_sign == 0:
            return None
        return round(abs(target - price) / pip_size, 1)

    # Read the forming candle separately; never mix it into indicators.
    candle_close_bias = "UNAVAILABLE - LIVE CANDLE NOT VERIFIED"
    candle_close_confidence = "none"
    candle_close_time = None

    if symbol.upper() == "XAUUSD":
        interval_map = {
            "1m": ("1m", 60),
            "5m": ("5m", 300),
            "15m": ("15m", 900),
            "30m": ("30m", 1800),
            "1h": ("1h", 3600),
        }

        if timeframe in interval_map:
            interval, candle_seconds = interval_map[timeframe]

            try:
                import json
                from urllib.request import Request, urlopen

                live_url = (
                    "https://biquote.io/api/XAUUSD/ohlc"
                    f"?interval={interval}&limit=10"
                )
                live_req = Request(
                    live_url,
                    headers={"User-Agent": "AdvancedTelegramScanner/1.0"},
                )

                with urlopen(live_req, timeout=15) as response:
                    live_payload = json.loads(
                        response.read().decode("utf-8")
                    )

                if live_payload.get("symbol", "").upper() != "XAUUSD":
                    raise ValueError("Unexpected symbol in live response")

                live_bars = live_payload.get("bars", [])
                open_bars = [
                    bar for bar in live_bars
                    if str(bar.get("isOpen", "")).strip().lower() == "true"
                ]

                if open_bars:
                    live_bar = max(
                        open_bars, key=lambda bar: bar.get("openTime", "")
                    )
                    live_stamp = pd.to_datetime(
                        live_bar["openTime"], utc=True, errors="raise"
                    )
                    age = (
                        pd.Timestamp.now(tz="UTC") - live_stamp
                    ).total_seconds()

                    if -60 <= age <= candle_seconds + 90:
                        live_open = float(live_bar["open"])
                        live_price = float(live_bar["close"])

                        if live_open > 0 and live_price > 0:
                            if live_price > live_open:
                                direction = "PROVISIONALLY BULLISH"
                            elif live_price < live_open:
                                direction = "PROVISIONALLY BEARISH"
                            else:
                                direction = "PROVISIONALLY FLAT"

                            candle_close_bias = (
                                f"{direction} — unfinished candle; "
                                "not a prediction of final close"
                            )
                            candle_close_confidence = (
                                "N/A — provisional direction only"
                            )
                            candle_close_time = (
                                live_stamp + pd.Timedelta(
                                    seconds=candle_seconds
                                )
                            ).isoformat()
            except Exception as exc:
                log.warning("Live candle check failed: %s", exc)

    # Approximate local support/resistance from recent candles.
    support = float(df["Low"].iloc[-22:-2].min())
    resistance = float(df["High"].iloc[-22:-2].max())

    # Display sensible decimals without assuming every symbol has 2.
    digits = 3 if symbol in ("XAUUSD", "XAGUSD", "USDJPY") else 5
    if symbol in ("BTCUSD", "ETHUSD"):
        digits = 2

    fmt = lambda x: f"{x:.{digits}f}"

    feed_name = (
        "BiQuote XAUUSD OHLC (reference feed; verify with broker)"
        if symbol.upper() == "XAUUSD"
        else f"Yahoo Finance ({SYMBOLS[symbol]})"
    )

    return {
        "symbol": symbol,
        "feed": feed_name,
        "tf": timeframe,
        "time": last_candle_time,
        "price": fmt(price),
        "previous_price": fmt(float(previous["Close"])),
        "movement": fmt(price - float(previous["Close"])),
        "movement_pct": round(
            (price / float(previous["Close"]) - 1) * 100, 4
        ) if float(previous["Close"]) != 0 else 0.0,
        "action": action,
        "score": score,
        "bias": bias,
        "forecast1": fmt(forecast_1),
        "forecast2": fmt(forecast_2),
        "forecast3": fmt(forecast_3),
        "forecast_move1": fmt(abs(forecast_1 - price)),
        "forecast_move2": fmt(abs(forecast_2 - price)),
        "forecast_move3": fmt(abs(forecast_3 - price)),
        "forecast_pips1": pip_move(forecast_1),
        "forecast_pips2": pip_move(forecast_2),
        "forecast_pips3": pip_move(forecast_3),
        "rsi": round(rsi, 1),
        "atr": fmt(atr),
        "green": round(green_pct, 1),
        "pressure_index": pressure_index,
        "pressure_label": pressure_label,
        "bullish_pressure_candles": bullish_candles,
        "bearish_pressure_candles": bearish_candles,
        "higher": higher_tf + ": " + htf_confirm,
        "candle_close_bias": candle_close_bias,
        "candle_close_confidence": candle_close_confidence,
        "candle_close_time": str(candle_close_time),
        "support": fmt(support),
        "resistance": fmt(resistance),
        "reasons": reasons,
        "entry": fmt(entry) if entry is not None else None,
        "zone": (
            f"{fmt(entry_low)} - {fmt(entry_high)}"
            if action != "WAIT" else None
        ),
        "sl": fmt(sl) if sl is not None else None,
        "tp1": fmt(tp1) if tp1 is not None else None,
        "tp2": fmt(tp2) if tp2 is not None else None,
        "tp3": fmt(tp3) if tp3 is not None else None,
    }


def format_signal(s):
    lines = [
        f"📊 {s['symbol']} | {s['tf']}",
        f"Signal: {s['action']} | Score: {s['score']}",
        f"Analysis candle close: {s['price']} (not a live broker quote)",
        f"Previous completed close: {s['previous_price']}",
        f"Last-candle move: {s['movement']} ({s['movement_pct']}%)",
        f"Data feed: {s['feed']}",
        f"Candle time: {s['time']}",
        f"RSI: {s['rsi']} | ATR: {s['atr']}",
        f"Green candles: {s['green']}%",
        f"Candle pressure index: {s['pressure_index']}/100 | {s['pressure_label']}",
        f"Pressure candles: bullish {s['bullish_pressure_candles']} / bearish {s['bearish_pressure_candles']}",
        "Note: estimated from candle OHLC; not real buyer/seller order-book data.",
        f"Trend confirmation: {s['higher']}",
        f"Current candle close forecast: {s['candle_close_bias']}",
        (
            "Forecast confidence: N/A — live candle not verified"
            if s.get("candle_close_confidence") == "none"
            else f"Forecast confidence: {s['candle_close_confidence']} (not guaranteed)"
        ),
        (
            f"Estimated candle close time: {s['candle_close_time']}"
            if s.get("candle_close_time")
            and str(s.get("candle_close_time")).strip().lower() not in ("none", "null", "")
            else "Estimated candle close time: unavailable — live candle not verified"
        ),
        f"Support: {s['support']}",
        f"Resistance: {s['resistance']}",
        "",
        "Reasons:",
    ]

    lines.extend([f"• {r}" for r in s["reasons"]])

    lines += [
        "",
        "DIRECTION FORECAST — NOT GUARANTEED",
        f"Bias: {s['bias']}",
        f"Scenario 1 target: {s['forecast1']} | price move: {s['forecast_move1']}",
        f"Scenario 2 target: {s['forecast2']} | price move: {s['forecast_move2']}",
        f"Scenario 3 target: {s['forecast3']} | price move: {s['forecast_move3']}",
    ]
    if s["forecast_pips1"] is not None:
        lines.append(
            f"Estimated FX pips: {s['forecast_pips1']} / "
            f"{s['forecast_pips2']} / {s['forecast_pips3']} "
            "(scenarios 1/2/3)"
        )
    else:
        lines.append("Pips not calculated for this symbol; verify broker contract size.")
    lines.append("ATR scenarios are estimates, not promises that price will reach them.")

    if s["action"] in ("BUY", "SELL"):
        lines += [
            "",
            f"Illustrative entry (candle-based; verify live broker Bid/Ask): {s['entry']}",
            f"Entry zone: {s['zone']}",
            f"Stop Loss: {s['sl']}",
            f"TP1: {s['tp1']}",
            f"TP2: {s['tp2']}",
            f"TP3: {s['tp3']}",
            "",
            "Risk levels are estimates; verify live broker price and spread.",
            "ALWAYS USE STOP LOSS (SL).",
        ]
    elif s["action"] in ("BUY SETUP", "SELL SETUP"):
        lines += [
            "",
            "⚠️ SETUP ONLY — not a confirmed entry.",
            "No entry, SL or TP issued. Wait for full confirmation.",
        ]
    else:
        lines += [
            "",
            "🔎 SETUP FORMING — not a confirmed entry.",
            f"Current score: {s['score']}.",
            f"Trend check: {s['higher']}.",
            f"RSI: {s['rsi']} | MACD/EMA confirmation is required.",
            "No entry, SL or TP is issued until the confirmation rules pass.",
        ]

    lines += [
        "",
        "ALWAYS USE STOP LOSS (SL).",
        "Check live MT5 Bid/Ask and spread before entering.",
    ]
    return "\n".join(lines)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await guard(update):
        return
    await update.effective_message.reply_text(
        "Advanced Telegram Scanner ready.\n\n"
        "/scan XAUUSD 5m\n"
        "/scan BTCUSD 15m\n"
        "/scanall\n"
        "/watch - automatic alerts ON\n"
        "/stop - automatic alerts OFF\n"
        "/status\n"
        "/markets"
    )


async def markets(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await guard(update):
        return
    await update.effective_message.reply_text(
        "Markets: " + ", ".join(SYMBOLS) +
        "\nTimeframes: " + ", ".join(TIMEFRAMES)
    )


async def scan_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await guard(update):
        return

    args = context.args
    symbol = args[0].upper() if args else "XAUUSD"
    timeframe = args[1].lower() if len(args) > 1 else "5m"

    if symbol not in SYMBOLS:
        await update.effective_message.reply_text(
            "Unknown symbol. Use /markets"
        )
        return
    if timeframe not in TIMEFRAMES:
        await update.effective_message.reply_text(
            "Invalid timeframe. Use /markets"
        )
        return

    await update.effective_message.reply_text(
        f"Scanning {symbol} ({timeframe})..."
    )

    try:
        result = await asyncio.to_thread(
            analyze, symbol, timeframe
        )
        await update.effective_message.reply_text(
            format_signal(result)
        )
    except Exception as exc:
        log.exception("Scan failed")
        await update.effective_message.reply_text(
            f"Scan failed: {type(exc).__name__}: {exc}\n"
            "Try 5m/15m or check internet and data availability."
        )


async def scanall(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await guard(update):
        return

    for symbol in SYMBOLS:
        try:
            result = await asyncio.to_thread(
                analyze, symbol, "5m"
            )
            await update.effective_message.reply_text(
                format_signal(result)
            )
        except Exception as exc:
            log.warning("%s scan failed: %s", symbol, exc)
            await update.effective_message.reply_text(
                f"{symbol}: data unavailable ({type(exc).__name__})."
            )


async def watch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await guard(update):
        return
    WATCHING.add(str(update.effective_chat.id))
    await update.effective_message.reply_text(
        "Automatic alerts ON for configured markets, 5m timeframe. "
        "New-candle signals only. Use /stop to disable."
    )


async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await guard(update):
        return
    WATCHING.discard(str(update.effective_chat.id))
    await update.effective_message.reply_text(
        "Automatic alerts OFF."
    )


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not await guard(update):
        return
    watching = str(update.effective_chat.id) in WATCHING
    await update.effective_message.reply_text(
        f"Bot running: yes\n"
        f"Auto alerts: {'ON' if watching else 'OFF'}\n"
        f"Interval: {INTERVAL_SECONDS} seconds\n"
        f"Markets: {len(SYMBOLS)}\n"
        f"Execution: signals only; no auto-trading"
    )


async def auto_scan(app: Application):
    while True:
        try:
            for chat_id in list(WATCHING):
                for symbol in SYMBOLS:
                    try:
                        result = await asyncio.to_thread(
                            analyze, symbol, "5m"
                        )

                        key = f"{chat_id}:{symbol}:5m"
                        candle = result["time"]

                        # Only alert once per completed candle.
                        if LAST_CANDLE.get(key) == candle:
                            continue
                        LAST_CANDLE[key] = candle

                        # Report each new completed candle, including WAIT.
                        await app.bot.send_message(
                            chat_id=int(chat_id),
                            text=format_signal(result),
                        )
                    except Exception as exc:
                        log.warning(
                            "Auto scan %s failed: %s", symbol, exc
                        )
        except Exception:
            log.exception("Auto scanner loop error")

        await asyncio.sleep(INTERVAL_SECONDS)


async def post_init(app: Application):
    asyncio.create_task(auto_scan(app), name="trader_star_auto_scan")




# Network workaround: prefer IPv4 DNS results on this device.
def _force_ipv4_dns_for_telegram():
    import socket
    original_getaddrinfo = socket.getaddrinfo

    def ipv4_first(host, port, family=0, type=0, proto=0, flags=0):
        results = original_getaddrinfo(host, port, family, type, proto, flags)
        ipv4 = [item for item in results if item[0] == socket.AF_INET]
        ipv6 = [item for item in results if item[0] == socket.AF_INET6]
        return ipv4 or results

    socket.getaddrinfo = ipv4_first

def main():
    if not TOKEN or not OWNER_CHAT_ID:
        raise SystemExit(
            "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env first."
        )

    # Force IPv4 DNS preference before the Telegram client starts.
    _force_ipv4_dns_for_telegram()

    request = HTTPXRequest(
        connection_pool_size=8,
        connect_timeout=90.0,
        read_timeout=90.0,
        write_timeout=90.0,
        pool_timeout=90.0,
    )

    app = (
        Application.builder()
        .token(TOKEN)
        .request(request)
        .get_updates_request(
            HTTPXRequest(
                connection_pool_size=2,
                connect_timeout=90.0,
                read_timeout=90.0,
                write_timeout=90.0,
                pool_timeout=90.0,
            )
        )
        .post_init(post_init)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("markets", markets))
    app.add_handler(CommandHandler("scan", scan_command))
    app.add_handler(CommandHandler("scanall", scanall))
    app.add_handler(CommandHandler("watch", watch))
    app.add_handler(CommandHandler("stop", stop))
    app.add_handler(CommandHandler("status", status))

    print("Advanced Telegram Scanner starting...")
    app.run_polling(drop_pending_updates=False)


if __name__ == "__main__":
    main()
