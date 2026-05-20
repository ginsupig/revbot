from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Optional
import hashlib
import logging

import pandas as pd

from .config import PerformanceConfig, ReversionConfig, RiskConfig
from .engine import ReversionEngine
from .ml import MLSignalLearner
from .performance import (
    EvalRecord,
    PerformanceTracker,
    TradeRecord,
    time_bucket_from_hour,
    utc_now_iso,
)
from .risk import RiskManager
from .trendfail import trend_fail_continue_strategy


class ReversionService:
    def __init__(
        self,
        strategy_config: ReversionConfig | None = None,
        risk_config: RiskConfig | None = None,
        performance_config: PerformanceConfig | None = None,
        log_file: str = "reversion_service.log",
    ) -> None:
        self.engine = ReversionEngine(strategy_config)
        self.risk = RiskManager(risk_config)
        self.ml_learner = MLSignalLearner()
        self.perf_cfg = performance_config or PerformanceConfig()
        self.perf = PerformanceTracker(self.perf_cfg.state_dir)

        self.ml_last_train_data_hash: Optional[str] = None
        self.ml_eval_count = 0
        self.ml_train_every = 25
        self.min_rows_for_ml = 120

        self.weights = {
            "mean_reversion": 0.30,
            "ml": 0.25,
            "trendfail": 0.15,
            "trend_following": 0.30,
        }
        self.min_trade_score = 0.36

        self.logger = logging.getLogger("ReversionService")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            fh = logging.FileHandler(log_file)
            ch = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    def evaluate_symbol(self, symbol: str, df: pd.DataFrame, account_equity: float) -> Dict[str, Any]:
        self.logger.info("Evaluating symbol=%s rows=%s", symbol, len(df))

        enriched = self.engine.calculate_indicators(df)
        decision = self.engine.get_decision(enriched, symbol=symbol)

        mean_reversion_score = self._score_mean_reversion(decision, enriched)
        ml_probability = self._get_ml_probability(df)
        trendfail_score = self._get_trendfail_score(df)
        trend_following_score = self._get_trend_following_score(enriched)

        component_scores = {
            "mean_reversion": round(mean_reversion_score, 4),
            "ml": round(ml_probability, 4),
            "trendfail": round(trendfail_score, 4),
            "trend_following": round(trend_following_score, 4),
        }

        regime = self._classify_regime(enriched)
        entry_style, router_reason = self._route_style(component_scores, regime, decision.signal)

        threshold = self.min_trade_score
        if self.perf_cfg.enable_adaptive_threshold:
            threshold = self.perf.suggest_threshold_adjustment(
                entry_style=entry_style,
                regime=regime,
                baseline_threshold=threshold,
                min_samples=self.perf_cfg.min_samples,
                max_adj=self.perf_cfg.max_threshold_adj,
            )

        weighted_score = self._weighted_score(component_scores)
        row = enriched.iloc[-1]

        hard_wait_reasons = {
            "Dollar_Volume_Too_Low",
            "Spread_Too_Wide",
            "Price_Too_Low",
        }
        if decision.reason in hard_wait_reasons:
            payload = {
                "symbol": symbol,
                "decision": decision.to_dict(),
                "component_scores": component_scores,
                "weights": self.weights.copy(),
                "trade_score": round(weighted_score, 4),
                "threshold": round(threshold, 4),
                "go_long": False,
                "regime": regime,
                "entry_style": entry_style,
                "router_reason": f"hard_gate:{decision.reason}",
            }
            self._log_eval(symbol, regime, decision, payload["router_reason"], component_scores, weighted_score, threshold, row)
            self._persist_eval(symbol, regime, entry_style, decision, payload["router_reason"], weighted_score, threshold, row, go_long=False)
            return payload

        go_long = weighted_score >= threshold and router_reason != "score_below_threshold"

        if entry_style == "trend_following" and component_scores["trend_following"] < 0.55:
            go_long = False
            router_reason = "trend_following_below_min"
        if entry_style == "trendfail" and component_scores["trendfail"] < 0.60:
            go_long = False
            router_reason = "trendfail_below_min"

        payload: Dict[str, Any] = {
            "symbol": symbol,
            "decision": decision.to_dict(),
            "component_scores": component_scores,
            "weights": self.weights.copy(),
            "trade_score": round(weighted_score, 4),
            "threshold": round(threshold, 4),
            "go_long": False,
            "regime": regime,
            "entry_style": entry_style,
            "router_reason": router_reason,
        }

        self._log_eval(symbol, regime, decision, router_reason, component_scores, weighted_score, threshold, row)
        self._persist_eval(symbol, regime, entry_style, decision, router_reason, weighted_score, threshold, row, go_long=go_long)

        if go_long:
            plan = self.risk.build_plan_for_style(
                entry_style=entry_style,
                account_equity=account_equity,
                decision=decision,
                conviction_score=weighted_score,
            )
            if plan is None:
                payload["router_reason"] = f"plan_rejected:{entry_style}"
                return payload

            payload["position_plan"] = asdict(plan)
            payload["go_long"] = True
            payload["portfolio_heat"] = float(plan.qty) * float(plan.risk_per_share)

            hour = datetime.utcnow().hour
            time_bucket = time_bucket_from_hour(hour)
            self.perf.log_trade(
                TradeRecord(
                    timestamp=utc_now_iso(),
                    symbol=symbol,
                    regime=regime,
                    entry_style=entry_style,
                    trade_score=round(weighted_score, 4),
                    threshold=round(threshold, 4),
                    entry_price=float(plan.entry_price),
                    stop_price=float(plan.stop_price),
                    target_price=float(plan.target_price),
                    qty=int(plan.qty),
                    rr_ratio=float(plan.rr_ratio),
                    risk_per_share=float(plan.risk_per_share),
                    reward_per_share=float(plan.reward_per_share),
                    position_value=float(plan.position_value),
                    time_bucket=time_bucket,
                )
            )
            self.logger.info(
                "symbol=%s regime=%s entry_style=%s router_reason=%s position_plan=%s",
                symbol, regime, entry_style, router_reason, plan,
            )

        return payload

    def recent_performance_summary(self) -> Dict[str, Any]:
        return self.perf.summarize_recent(limit=500)

    def _log_eval(self, symbol, regime, decision, router_reason, component_scores, weighted_score, threshold, row):
        self.logger.info(
            "symbol=%s regime=%s decision=%s reason=%s router_reason=%s component_scores=%s trade_score=%.4f threshold=%.4f close=%.2f ri=%.4f rsi=%.4f adx=%.4f",
            symbol,
            regime,
            decision.signal,
            decision.reason,
            router_reason,
            component_scores,
            weighted_score,
            threshold,
            float(row["close"]) if pd.notna(row["close"]) else 0.0,
            float(row["ri"]) if pd.notna(row["ri"]) else 0.0,
            float(row["rsi"]) if pd.notna(row["rsi"]) else 0.0,
            float(row["adx"]) if pd.notna(row["adx"]) else 0.0,
        )

    def _persist_eval(self, symbol, regime, entry_style, decision, router_reason, weighted_score, threshold, row, go_long: bool = False):
        hour = datetime.utcnow().hour
        time_bucket = time_bucket_from_hour(hour)
        self.perf.log_evaluation(
            EvalRecord(
                timestamp=utc_now_iso(),
                symbol=symbol,
                regime=regime,
                entry_style=entry_style,
                decision=decision.signal,
                reason=decision.reason,
                router_reason=router_reason,
                trade_score=round(weighted_score, 4),
                threshold=round(threshold, 4),
                go_long=go_long,
                close=float(row["close"]) if pd.notna(row["close"]) else None,
                ri=float(row["ri"]) if pd.notna(row["ri"]) else None,
                rsi=float(row["rsi"]) if pd.notna(row["rsi"]) else None,
                adx=float(row["adx"]) if pd.notna(row["adx"]) else None,
                time_bucket=time_bucket,
            )
        )

    def _classify_regime(self, enriched: pd.DataFrame) -> str:
        row = enriched.iloc[-1]
        adx = float(row["adx"]) if pd.notna(row["adx"]) else 0.0
        trend_signal = int(row["trend_following_signal"]) if "trend_following_signal" in enriched.columns and pd.notna(row["trend_following_signal"]) else 0
        hard_max = getattr(self.engine.config, "adx_hard_max", 50.0)
        if adx >= hard_max:
            return "trend"
        if adx >= 25 and trend_signal == 1:
            return "trend"
        if adx < 15:
            return "range"
        return "reversion"

    def _route_style(self, component_scores: Dict[str, float], regime: str, engine_signal: str) -> tuple[str, str]:
        mr = float(component_scores["mean_reversion"])
        tf = float(component_scores["trend_following"])
        tff = float(component_scores["trendfail"])

        if regime == "trend" and tf >= 0.55:
            return "trend_following", "trend_regime_alignment"
        if engine_signal == "LONG_REVERSION" and mr >= 0.45:
            return "mean_reversion", "engine_validated_reversion"
        if tff >= 0.60:
            return "trendfail", "trendfail_signal_strength"

        best_style = max(
            ("mean_reversion", mr),
            ("trend_following", tf),
            ("trendfail", tff),
            key=lambda x: x[1],
        )
        if best_style[1] < self.min_trade_score:
            return best_style[0], "score_below_threshold"
        if best_style[0] == "trend_following":
            return best_style[0], "best_style_edge_trend_following"
        if best_style[0] == "trendfail":
            return best_style[0], "best_style_edge_trendfail"
        return best_style[0], "best_style_edge_mean_reversion"

    def _score_mean_reversion(self, decision, enriched: pd.DataFrame) -> float:
        score = 0.20
        if decision.signal == "LONG_REVERSION":
            score += 0.25
        if decision.close is not None and decision.lb1 is not None and decision.lb2 is not None:
            zone_width = max(decision.lb1 - decision.lb2, 1e-9)
            depth = (decision.lb1 - decision.close) / zone_width
            score += min(max(depth, 0.0), 1.0) * 0.20
        if decision.ri is not None:
            threshold = getattr(self.engine.config, "ri_threshold", -0.5)
            oversold_distance = max(threshold - decision.ri, 0.0)
            score += min(oversold_distance / 0.75, 1.0) * 0.15
        if decision.rsi is not None:
            softness = max((self.engine.config.rsi_max + 10) - decision.rsi, 0.0)
            score += min(softness / 25.0, 1.0) * 0.10
        if decision.adx is not None:
            adx_buffer = max((self.engine.config.adx_max + 5) - decision.adx, 0.0)
            score += min(adx_buffer / 20.0, 1.0) * 0.10
        return max(0.0, min(score, 1.0))

    def _get_ml_probability(self, df: pd.DataFrame) -> float:
        self.ml_eval_count += 1
        if len(df) < self.min_rows_for_ml:
            return 0.5

        data_hash = self._hash_frame_tail(df, rows=200)
        should_train = (
            self.ml_last_train_data_hash != data_hash
            and (self.ml_eval_count == 1 or self.ml_eval_count % self.ml_train_every == 0)
        )
        if should_train:
            try:
                X, y = self.ml_learner.prepare_features(df)
                if len(X) >= 50 and y.nunique() >= 2:
                    self.ml_learner.model.fit(X, y)
                    self.ml_last_train_data_hash = data_hash
                    self.logger.info("ML retrained rows=%s classes=%s", len(X), int(y.nunique()))
            except Exception as exc:
                self.logger.warning("ML retrain error: %s", exc)

        try:
            X, _ = self.ml_learner.prepare_features(df)
            if len(X) == 0:
                return 0.5
            proba = self.ml_learner.model.predict_proba(X)
            if len(proba) == 0:
                return 0.5
            row = proba[-1]
            if len(row) < 2:
                return 0.5
            return float(row[1])
        except Exception as exc:
            self.logger.warning("ML probability error: %s", exc)
            return 0.5

    def _get_trendfail_score(self, df: pd.DataFrame) -> float:
        try:
            tf_out = trend_fail_continue_strategy(
                df,
                window=getattr(self.engine.config, "trendfail_window", 20),
                threshold=getattr(self.engine.config, "trendfail_threshold", 0.005),
            )
            signal = int(tf_out["signal"].iloc[-1])
            if signal == 1:
                return 0.75
            if signal == -1:
                return 0.10
            return 0.45
        except Exception as exc:
            self.logger.warning("Trend-fail signal error: %s", exc)
            return 0.45

    def _get_trend_following_score(self, enriched: pd.DataFrame) -> float:
        try:
            row = enriched.iloc[-1]
            signal = int(row["trend_following_signal"]) if "trend_following_signal" in enriched.columns and pd.notna(row["trend_following_signal"]) else 0
            base = 0.30
            if signal == 0:
                return base
            macd_line = float(row["macd_line"]) if pd.notna(row["macd_line"]) else 0.0
            macd_signal = float(row["macd_signal"]) if pd.notna(row["macd_signal"]) else 0.0
            close = float(row["close"]) if pd.notna(row["close"]) else 0.0
            trend_ema = float(row["trend_ema"]) if pd.notna(row["trend_ema"]) else close
            macd_strength = min(max(macd_line - macd_signal, 0.0), 1.0)
            ema_distance = min(max((close - trend_ema) / max(close, 1e-9), 0.0) * 20.0, 1.0)
            return max(0.0, min(0.45 + 0.25 * macd_strength + 0.30 * ema_distance, 1.0))
        except Exception as exc:
            self.logger.warning("Trend-following score error: %s", exc)
            return 0.30

    def _weighted_score(self, component_scores: Dict[str, float]) -> float:
        return sum(self.weights.get(name, 0.0) * float(value) for name, value in component_scores.items())

    @staticmethod
    def _hash_frame_tail(df: pd.DataFrame, rows: int = 200) -> str:
        tail = df.tail(rows).copy()
        payload = tail.to_csv(index=True).encode("utf-8", errors="ignore")
        return hashlib.sha256(payload).hexdigest()
