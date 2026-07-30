from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

from .config import ReversionConfig
from .indicators import (
    calculate_adx,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_momentum,
    calculate_rsi,
    calculate_volatility,
    calculate_vwap,
    normalized_reversion_index,
)
from .models import ReversionDecision, SafetyDecision


class ReversionEngine:
    REQUIRED_COLUMNS: Iterable[str] = ("open", "high", "low", "close", "volume")

    def __init__(self, config: ReversionConfig | None = None) -> None:
        self.config = config or ReversionConfig()

    def validate_input(self, df: pd.DataFrame) -> None:
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        if len(df) < self.config.min_history:
            raise ValueError(
                f"Not enough rows. Need at least {self.config.min_history}, got {len(df)}."
            )

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_input(df)

        out = df.copy()
        close = out["close"].astype(float)

        sma = close.rolling(
            self.config.band_length,
            min_periods=self.config.band_length,
        ).mean()
        std = close.rolling(
            self.config.band_length,
            min_periods=self.config.band_length,
        ).std(ddof=0)

        out["sma"] = sma
        out["ub1"] = sma + (self.config.band_std_1 * std)
        out["ub2"] = sma + (self.config.band_std_2 * std)
        out["lb1"] = sma - (self.config.band_std_1 * std)
        out["lb2"] = sma - (self.config.band_std_2 * std)

        out["ri"] = normalized_reversion_index(close, self.config.ri_length)
        out["rsi"] = calculate_rsi(out, self.config.rsi_length)
        out["adx"] = calculate_adx(out, self.config.adx_length)
        out["atr"] = calculate_atr(out, self.config.atr_length)
        out["vwap"] = calculate_vwap(out)
        out["trend_ema"] = calculate_ema(close, self.config.trend_ema_length)

        out["avg_volume"] = out["volume"].astype(float).rolling(
            self.config.volume_lookback,
            min_periods=self.config.volume_lookback,
        ).mean()

        out["dollar_volume"] = out["close"].astype(float) * out["volume"].astype(float)
        out["avg_dollar_volume"] = out["dollar_volume"].rolling(
            self.config.volume_lookback,
            min_periods=self.config.volume_lookback,
        ).mean()

        # Bar fetches return OHLCV only, so this column is all-NaN in the live
        # path and the Spread_Too_Wide check below always fails open. The real
        # spread limit is enforced at submit time from the live quote
        # (main._spread_blocks_entry); the column stays so a caller that DOES
        # supply per-bar spreads (tests, a future quote-enriched fetch) is still
        # honored here.
        out["spread_bps"] = out.get("spread_bps", pd.Series(index=out.index, dtype=float))

        bb_sma, bb_upper, bb_lower = calculate_bollinger_bands(
            out, length=self.config.band_length, num_std=2.0
        )
        out["bb_sma"] = bb_sma
        out["bb_upper"] = bb_upper
        out["bb_lower"] = bb_lower

        macd_line, macd_signal, macd_hist = calculate_macd(out)
        out["macd_line"] = macd_line
        out["macd_signal"] = macd_signal
        out["macd_hist"] = macd_hist

        out["momentum"] = calculate_momentum(out)
        out["volatility"] = calculate_volatility(out)

        out["trend_following_signal"] = (
            (out["close"] > out["trend_ema"]) & (out["macd_line"] > out["macd_signal"])
        ).astype(int)

        return out

    def is_market_safe(self, df: pd.DataFrame) -> SafetyDecision:
        row = df.iloc[-1]

        adx = float(row["adx"]) if pd.notna(row["adx"]) else None
        rsi = float(row["rsi"]) if pd.notna(row["rsi"]) else None
        spread_bps = (
            float(row["spread_bps"])
            if "spread_bps" in df.columns and pd.notna(row["spread_bps"])
            else None
        )
        dollar_volume = (
            float(row["avg_dollar_volume"]) if pd.notna(row["avg_dollar_volume"]) else None
        )
        close = float(row["close"]) if pd.notna(row["close"]) else None

        if adx is None or rsi is None or close is None:
            return SafetyDecision(False, "Indicators_Not_Ready", adx=adx, rsi=rsi)

        if close < self.config.min_price:
            return SafetyDecision(
                False,
                "Price_Too_Low",
                adx=adx,
                rsi=rsi,
                spread_bps=spread_bps,
                dollar_volume=dollar_volume,
            )

        if dollar_volume is not None and dollar_volume < self.config.min_dollar_volume:
            return SafetyDecision(
                False,
                "Dollar_Volume_Too_Low",
                adx=adx,
                rsi=rsi,
                spread_bps=spread_bps,
                dollar_volume=dollar_volume,
            )

        if spread_bps is not None and spread_bps > self.config.max_spread_bps:
            return SafetyDecision(
                False,
                "Spread_Too_Wide",
                adx=adx,
                rsi=rsi,
                spread_bps=spread_bps,
                dollar_volume=dollar_volume,
            )

        if adx is not None and adx >= self.config.adx_hard_max and rsi >= self.config.rsi_hard_max:
            return SafetyDecision(
                False,
                "Momentum_Too_Extended",
                adx=adx,
                rsi=rsi,
                spread_bps=spread_bps,
                dollar_volume=dollar_volume,
            )

        # Mirror of the blow-off-top block above: refuse to fade a strong
        # *downtrend* that is also deeply oversold (a falling knife). Without
        # this, a long-reversion would buy names like ADX~60 / RSI~22.
        if adx is not None and adx >= self.config.adx_hard_max and rsi <= self.config.rsi_hard_min:
            return SafetyDecision(
                False,
                "Downtrend_Too_Extended",
                adx=adx,
                rsi=rsi,
                spread_bps=spread_bps,
                dollar_volume=dollar_volume,
            )

        return SafetyDecision(
            True,
            "Safe",
            adx=adx,
            rsi=rsi,
            spread_bps=spread_bps,
            dollar_volume=dollar_volume,
        )

    def _check_confirmations(
        self,
        *,
        close: float,
        prev,
        row,
        config,
        direction: str = "long"
    ) -> tuple[bool, bool, bool, bool]:
        """Check reclaim, volume, vwap, trend confirmations for entry direction.

        Args:
            direction: "long" or "short" — determines which confirmations apply

        Returns:
            (reclaim_ok, volume_ok, vwap_ok, trend_ok)
        """
        reclaim_ok = True
        volume_ok = True
        vwap_ok = True
        trend_ok = True

        if prev is not None:
            avg_volume = float(row["avg_volume"]) if pd.notna(row["avg_volume"]) else 0.0
            volume_ok = (
                avg_volume <= 0
                or float(row["volume"]) >= avg_volume * config.volume_multiplier_min
            )

            # NOTE ON NAMING: this is an *up-tick off the band*, not a full band
            # reclaim. get_decision only reaches here when the current close is
            # still INSIDE the reversion zone (close <= lb1 for longs), so a
            # literal reclaim (close back above lb1) is unreachable by
            # construction — the old first branch `prev_close <= prev_lb1 and
            # close > lb1` could never be true and the second branch was doing
            # all the work. Expressed directly so the rule matches its docs:
            # the prior close was at/below the band and price is ticking up.
            if direction == "long":
                prev_close = float(prev["close"])
                prev_lb1 = float(prev["lb1"]) if pd.notna(prev["lb1"]) else float(row["lb1"])
                reclaim_ok = prev_close <= prev_lb1 and close > prev_close
            else:  # short
                prev_close = float(prev["close"])
                prev_ub1 = float(prev["ub1"]) if pd.notna(prev["ub1"]) else float(row["ub1"])
                reclaim_ok = prev_close >= prev_ub1 and close < prev_close

        # VWAP filter (same for both directions)
        vwap = float(row["vwap"]) if pd.notna(row["vwap"]) else None
        if config.use_vwap_filter and vwap and vwap > 0:
            vwap_ok = abs((close - vwap) / vwap) <= config.max_vwap_extension_pct

        # Trend filter (mirrored for direction)
        if config.use_trend_filter:
            trend_ema = float(row["trend_ema"]) if pd.notna(row["trend_ema"]) else close
            if direction == "long":
                trend_ok = close >= trend_ema * (1.0 - config.trend_filter_band_pct)
            else:  # short
                trend_ok = close <= trend_ema * (1.0 + config.trend_filter_band_pct)

        return reclaim_ok, volume_ok, vwap_ok, trend_ok

    def get_decision(self, df: pd.DataFrame, symbol: str | None = None, short_bias: bool = False) -> ReversionDecision:
        row = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else None

        needed = ["close", "lb1", "lb2", "sma", "ri", "rsi", "adx", "atr", "vwap"]
        if any(pd.isna(row[col]) for col in needed):
            return ReversionDecision(signal="WAIT", reason="Indicators_Not_Ready", symbol=symbol)

        safety = self.is_market_safe(df)
        if not safety.is_safe:
            return ReversionDecision(
                signal="WAIT",
                reason=safety.reason,
                symbol=symbol,
                close=float(row["close"]),
                lb1=float(row["lb1"]),
                lb2=float(row["lb2"]),
                sma=float(row["sma"]),
                ri=float(row["ri"]),
                rsi=safety.rsi,
                adx=safety.adx,
                atr=float(row["atr"]),
                vwap=float(row["vwap"]),
                spread_bps=safety.spread_bps,
                dollar_volume=safety.dollar_volume,
            )

        close = float(row["close"])
        lb1 = float(row["lb1"])
        lb2 = float(row["lb2"])
        ub1 = float(row["ub1"]) if pd.notna(row["ub1"]) else None
        ub2 = float(row["ub2"]) if pd.notna(row["ub2"]) else None
        sma = float(row["sma"])
        ri = float(row["ri"])
        atr = float(row["atr"])
        vwap = float(row["vwap"])
        current_rsi = float(row["rsi"])
        trend_ema = float(row["trend_ema"]) if pd.notna(row["trend_ema"]) else close

        if atr <= 0:
            return ReversionDecision(
                signal="WAIT",
                reason="ATR_Invalid",
                symbol=symbol,
                close=close,
                lb1=lb1,
                lb2=lb2,
                sma=sma,
                ri=ri,
                rsi=current_rsi,
                adx=float(row["adx"]),
                atr=atr,
                vwap=vwap,
                spread_bps=safety.spread_bps,
                dollar_volume=safety.dollar_volume,
            )

        # Price must be at or below lb1 (or inside the lb2–lb1 band).
        in_reversion_zone = close <= lb1

        # Oversold gate: configurable AND/OR. AND requires both momentum
        # exhaustion (RI) and oscillator confirmation (RSI), filtering weak
        # dips that are trend continuations. OR fires on ~50% of bars with
        # rsi_max=48 and was the root cause of entries in strong trends.
        ri_oversold = ri <= self.config.ri_threshold
        rsi_oversold = current_rsi <= self.config.rsi_max
        # Normalized here too: per-symbol configs and research harnesses build
        # ReversionConfig directly, and an unnormalized "OR" silently selecting
        # the AND gate is invisible in the logs.
        gate_mode = str(self.config.oversold_gate or "and").strip().lower()
        if gate_mode not in ("and", "or"):
            logging.warning(
                "oversold_gate=%r is not 'and'/'or' — using the AND gate.",
                self.config.oversold_gate,
            )
            gate_mode = "and"
        if gate_mode == "or":
            oversold = ri_oversold or rsi_oversold
        else:
            oversold = ri_oversold and rsi_oversold

        bullish_close = True
        if prev is not None:
            bullish_close = close >= float(row["open"])

        # Use shared confirmation logic for reclaim, volume, vwap, trend.
        reclaim_lb1, volume_ok, vwap_ok, trend_ok = self._check_confirmations(
            close=close,
            prev=prev,
            row=row,
            config=self.config,
            direction="long"
        )

        # ADX trend-strength filter: reject long reversion entries when the
        # trend is strong (ADX > adx_max). Mean-reversion in a strong trend is
        # the #1 way to lose money — routine pullbacks in trends are
        # continuations, not reversions. Configurable via ADX_MAX env var.
        adx_val = float(row["adx"])
        adx_too_strong = adx_val > self.config.adx_max

        if not in_reversion_zone:
            reason = "Not_In_Reversion_Zone"
        elif adx_too_strong:
            reason = "ADX_Trend_Too_Strong"
        elif not oversold:
            reason = "RI_Not_Oversold"
        elif self.config.require_reclaim_lb1 and not reclaim_lb1:
            reason = "No_Reclaim_Trigger"
        elif self.config.require_bullish_close and not bullish_close:
            reason = "No_Bullish_Confirmation"
        elif self.config.require_volume_expansion and not volume_ok:
            reason = "No_Volume_Expansion"
        elif not vwap_ok:
            reason = "VWAP_Extension_Too_Far"
        elif not trend_ok:
            reason = "Higher_Timeframe_Trend_Too_Weak"
        else:
            return ReversionDecision(
                signal="LONG_REVERSION",
                reason="Validated_Long_Reversion",
                symbol=symbol,
                close=close,
                lb1=lb1,
                lb2=lb2,
                ub1=ub1,
                ub2=ub2,
                sma=sma,
                ri=ri,
                rsi=current_rsi,
                adx=float(row["adx"]),
                atr=atr,
                vwap=vwap,
                spread_bps=safety.spread_bps,
                dollar_volume=safety.dollar_volume,
            )

        # Long setup did not validate. The long and short zones are mutually
        # exclusive (price can't be at/below lb1 and at/above ub1 at once), so
        # we only reach a short setup when there was no long to begin with.
        if self.config.enable_shorts and ub1 is not None and ub2 is not None:
            short_decision = self._evaluate_short(
                row=row,
                prev=prev,
                safety=safety,
                symbol=symbol,
                close=close,
                lb1=lb1,
                lb2=lb2,
                ub1=ub1,
                ub2=ub2,
                sma=sma,
                ri=ri,
                current_rsi=current_rsi,
                atr=atr,
                vwap=vwap,
                trend_ema=trend_ema,
                short_bias=short_bias,
            )
            if short_decision is not None:
                return short_decision

        return ReversionDecision(
            signal="WAIT",
            reason=reason,
            symbol=symbol,
            close=close,
            lb1=lb1,
            lb2=lb2,
            ub1=ub1,
            ub2=ub2,
            sma=sma,
            ri=ri,
            rsi=current_rsi,
            adx=float(row["adx"]),
            atr=atr,
            vwap=vwap,
            spread_bps=safety.spread_bps,
            dollar_volume=safety.dollar_volume,
        )

    def _evaluate_short(
        self,
        *,
        row,
        prev,
        safety,
        symbol,
        close: float,
        lb1: float,
        lb2: float,
        ub1: float,
        ub2: float,
        sma: float,
        ri: float,
        current_rsi: float,
        atr: float,
        vwap: float,
        trend_ema: float,
        short_bias: bool = False,
    ) -> ReversionDecision | None:
        """Mirror of the long reversion setup: short overbought rips.

        Returns a SHORT_REVERSION decision when every mirrored gate passes,
        otherwise None (the caller falls through to the long-side WAIT reason).
        All filter flags (reclaim/bullish-close/volume/vwap/trend) reuse the
        same config switches as the long side, applied in the short direction.

        short_bias relaxes the overbought thresholds (used when the market is
        risk-off and favor-shorts mode is on) so the bot fades rips more eagerly
        in a downtrend.
        """
        adx = float(row["adx"])

        # Reflected hard guard: don't short a capitulation that is also strongly
        # trending down (mirror of the long blow-off-top block in is_market_safe).
        if adx >= self.config.adx_hard_max and current_rsi <= self.config.rsi_hard_min:
            return None

        # ADX trend-strength veto — the exact mirror of the long side's
        # ADX_Trend_Too_Strong gate. Shorting an "overbought rip" inside a
        # strong trend (ADX 40-50 with RSI high) is fading a breakout, the #1
        # way the short side bleeds; the long side already refused entries at
        # this trend strength while the short side had no equivalent check.
        if adx > self.config.adx_max:
            return None

        # Price must be at or above ub1 (or inside the ub1–ub2 band).
        in_short_zone = close >= ub1
        if not in_short_zone:
            return None

        rsi_min = self.config.risk_off_rsi_min if short_bias else self.config.rsi_min
        ri_short_threshold = (
            self.config.risk_off_ri_short_threshold if short_bias else self.config.ri_short_threshold
        )
        # AND gate: require BOTH momentum exhaustion (RI overbought) AND
        # oscillator confirmation (RSI overbought) — mirrors the long side.
        overbought = (
            ri >= ri_short_threshold
            and current_rsi >= rsi_min
        )
        if not overbought:
            return None

        bearish_close = True
        if prev is not None:
            bearish_close = close <= float(row["open"])

        # Use shared confirmation logic for reclaim, volume, vwap, trend (short version).
        reject_ub1, volume_ok, vwap_ok, trend_ok = self._check_confirmations(
            close=close,
            prev=prev,
            row=row,
            config=self.config,
            direction="short"
        )

        if self.config.require_reclaim_lb1 and not reject_ub1:
            return None
        if self.config.require_bullish_close and not bearish_close:
            return None
        if self.config.require_volume_expansion and not volume_ok:
            return None
        if not vwap_ok or not trend_ok:
            return None

        return ReversionDecision(
            signal="SHORT_REVERSION",
            reason="Validated_Short_Reversion",
            symbol=symbol,
            close=close,
            lb1=lb1,
            lb2=lb2,
            ub1=ub1,
            ub2=ub2,
            sma=sma,
            ri=ri,
            rsi=current_rsi,
            adx=adx,
            atr=atr,
            vwap=vwap,
            spread_bps=safety.spread_bps,
            dollar_volume=safety.dollar_volume,
        )
