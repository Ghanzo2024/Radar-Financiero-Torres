"""
RADAR FINANCIERO 2026 — MOTOR 2: ESTRATEGIAS Y CANDIDATOS v1.0
Fase 9 del Roadmap (Tareas 65 a 71)
"""
from typing import Tuple, Dict, Any
from pathlib import Path
import pandas as pd
import numpy as np
import json

def run_motor2_strategies(scanner_df: pd.DataFrame, capital: float = 20000.0, risk_pct: float = 0.01) -> Tuple[pd.DataFrame, dict]:
    if scanner_df is None or scanner_df.empty:
        return pd.DataFrame(), {}

    df = scanner_df.copy()
    candidates = []
    dollar_risk = capital * risk_pct

    for _, row in df.iterrows():
        t = row["ticker"]
        close = float(row["close"])
        stage = str(row["stage"])
        dist_p55 = float(row.get("distance_to_prior55_high_pct", 0.0))
        vol_ratio = float(row.get("volume_ratio_20d", 1.0))
        confluence = int(row.get("confluence", 0))
        mom = float(row.get("momentum_score", 50.0))
        rs = float(row.get("rs_score", 50.0))
        sma20 = float(row.get("sma20", close * 0.95))
        sma50 = float(row.get("sma50", close * 0.90))
        stop_p = float(row.get("stop_price", close * 0.93))

        # 1. BREAKOUT SWING (T65)
        is_breakout_cand = ("STAGE 2" in stage and dist_p55 >= -2.0 and vol_ratio >= 1.15 and confluence >= 4)
        if is_breakout_cand:
            candidates.append({
                "ticker": t,
                "strategy": "BREAKOUT SWING",
                "horizon": "Swing (3–15 días)",
                "close": close,
                "stage": stage,
                "score": float(row.get("score_swing", 70.0)),
                "entry_trigger": f"Ruptura sobre ${max(close, float(row.get('prior_55d_high', close))):.2f}",
                "stop_price": stop_p,
                "risk_pct": float(row.get("risk_distance_pct", 5.0)),
                "confluence": confluence,
                "reason": f"Consolidación ajustada a {dist_p55:.1f}% del pivote con volumen relativo {vol_ratio:.2f}x."
            })

        # 2. PULLBACK SWING (T66)
        dist_sma20 = abs(close / sma20 - 1.0) * 100.0 if sma20 > 0 else 10.0
        is_pullback_cand = ("STAGE 2" in stage and dist_sma20 <= 2.5 and close >= sma20 * 0.98 and vol_ratio <= 1.05 and confluence >= 4)
        if is_pullback_cand and not is_breakout_cand:
            candidates.append({
                "ticker": t,
                "strategy": "PULLBACK SWING",
                "horizon": "Swing (5–20 días)",
                "close": close,
                "stage": stage,
                "score": float(row.get("score_swing", 70.0)),
                "entry_trigger": f"Rebote en soporte SMA20 (${sma20:.2f}) con giro de vela",
                "stop_price": round(sma20 - float(row.get("atr14", 1.0)) * 0.75, 2),
                "risk_pct": float(row.get("risk_distance_pct", 5.0)),
                "confluence": confluence,
                "reason": f"Retroceso ordenado a media de 20 días con volumen en secado ({vol_ratio:.2f}x)."
            })

        # 3. MOMENTUM POSITION (T68)
        is_mom_pos = ("STAGE 2" in stage and rs >= 75.0 and mom >= 75.0 and confluence >= 5)
        if is_mom_pos:
            candidates.append({
                "ticker": t,
                "strategy": "MOMENTUM POSITION",
                "horizon": "Position (4–16 semanas)",
                "close": close,
                "stage": stage,
                "score": float(row.get("score_position", 70.0)),
                "entry_trigger": f"Continuidad tendencial sobre SMA50 (${sma50:.2f})",
                "stop_price": round(sma50 * 0.95, 2),
                "risk_pct": round(abs(close - sma50 * 0.95) / close * 100.0, 2),
                "confluence": confluence,
                "reason": f"Liderazgo relativo persistente (RS: {rs:.1f}, Momentum: {mom:.1f})."
            })

    cand_df = pd.DataFrame(candidates)
    if not cand_df.empty:
        cand_df["risk_per_share"] = np.maximum(cand_df["close"] - cand_df["stop_price"], 0.01)
        cand_df["suggested_shares"] = np.floor(dollar_risk / cand_df["risk_per_share"]).astype(int)
        cand_df["position_value"] = (cand_df["suggested_shares"] * cand_df["close"]).round(2)
        cand_df = cand_df.sort_values(["confluence", "score"], ascending=False).reset_index(drop=True)

    summary = {
        "total_candidates": len(cand_df),
        "breakout_swing_count": int((cand_df["strategy"] == "BREAKOUT SWING").sum()) if not cand_df.empty else 0,
        "pullback_swing_count": int((cand_df["strategy"] == "PULLBACK SWING").sum()) if not cand_df.empty else 0,
        "momentum_position_count": int((cand_df["strategy"] == "MOMENTUM POSITION").sum()) if not cand_df.empty else 0,
    }
    return cand_df, summary
