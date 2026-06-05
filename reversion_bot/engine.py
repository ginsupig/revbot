from __future__ import annotations

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
        out["atr"] = calculate_atr(out, 14)
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

        return SafetyDecision(
            True,
            "Safe",
            adx=adx,
            rsi=rsi,
            spread_bps=spread_bps,
            dollar_volume=dollar_volume,
        )

    def get_decision(self, df: pd.DataFrame, symbol: str | None = None) -> ReversionDecision:
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

        oversold = (
            ri <= self.config.ri_threshold
            or current_rsi <= self.config.rsi_max
        )

        reclaim_lb1 = True
        bullish_close = True
        volume_ok = True
        vwap_ok = True
        trend_ok = True

        if prev is not None:
            prev_close = float(prev["close"])
            prev_lb1 = float(prev["lb1"]) if pd.notna(prev["lb1"]) else lb1
            reclaim_lb1 = (prev_close < prev_lb1 and close > lb1) or close > prev_close
            bullish_close = close >= float(row["open"])

            avg_volume = float(row["avg_volume"]) if pd.notna(row["avg_volume"]) else 0.0
            volume_ok = (
                avg_volume <= 0
                or float(row["volume"]) >= avg_volume * self.config.volume_multiplier_min
            )

        if self.config.use_vwap_filter and vwap > 0:
            vwap_ok = abs((close - vwap) / vwap) <= self.config.max_vwap_extension_pct

        if self.config.use_trend_filter:
            trend_ok = close >= trend_ema * 0.965

        if not in_reversion_zone:
            reason = "Not_In_Reversion_Zone"
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
    ) -> ReversionDecision | None:
        """Mirror of the long reversion setup: short overbought rips.

        Returns a SHORT_REVERSION decision when every mirrored gate passes,
        otherwise None (the caller falls through to the long-side WAIT reason).
        All filter flags (reclaim/bullish-close/volume/vwap/trend) reuse the
        same config switches as the long side, applied in the short direction.
        """
        adx = float(row["adx"])

        # Reflected hard guard: don't short a capitulation that is also strongly
        # trending down (mirror of the long blow-off-top block in is_market_safe).
        if adx >= self.config.adx_hard_max and current_rsi <= self.config.rsi_hard_min:
            return None

        # Price must be at or above ub1 (or inside the ub1–ub2 band).
        in_short_zone = close >= ub1
        if not in_short_zone:
            return None

        overbought = (
            ri >= self.config.ri_short_threshold
            or current_rsi >= self.config.rsi_min
        )
        if not overbought:
            return None

        reject_ub1 = True
        bearish_close = True
        volume_ok = True
        vwap_ok = True
        trend_ok = True

        if prev is not None:
            prev_close = float(prev["close"])
            prev_ub1 = float(prev["ub1"]) if pd.notna(prev["ub1"]) else ub1
            # Mirror of reclaim_lb1: a rejection back below ub1, or any down-tick.
            reject_ub1 = (prev_close > prev_ub1 and close < ub1) or close < prev_close
            bearish_close = close <= float(row["open"])

            avg_volume = float(row["avg_volume"]) if pd.notna(row["avg_volume"]) else 0.0
            volume_ok = (
                avg_volume <= 0
                or float(row["volume"]) >= avg_volume * self.config.volume_multiplier_min
            )

        if self.config.use_vwap_filter and vwap > 0:
            vwap_ok = abs((close - vwap) / vwap) <= self.config.max_vwap_extension_pct

        if self.config.use_trend_filter:
            # Mirror of the long trend filter: only short when not far *above* the
            # higher-timeframe trend (i.e. price is not in a strong uptrend).
            trend_ok = close <= trend_ema * 1.035

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
