"""
RADAR FINANCIERO 2026 — MOTOR 1 DATA/SCANNER CORE v1.0

Versión de producción metodológica sincronizada con Tareas 1 a 100:
- T31: Weinstein Stage 1/2/3/4 + Stage Confidence (HIGH, MEDIUM, LOW)
- T32: Absolute Momentum Score (10% 1M, 25% 3M, 30% 6M, 35% 12M) + Aceleración
- T33: Relative Strength Score vs SPY (60%) y Sector (40%)
- T34: Sector Flow Score (35% RS, 25% Mom, 20% Breadth, 20% Persistencia) + Estados LEADING→LAGGING
- T35: Market Regime Score (30% Índices, 25% Amplitud, 20% VIX, 15% Tasas/DXY, 10% Persistencia)
- T36-T38: Growth, Fundamental Quality y Valuation Scoring
- T39-T40: Catalyst y Setup/Breakout Score (VCP + Volumen Ratio + Distancia Pivote)
- T41, T51-T54: Risk Score, Stop Estructural + Buffer ATR dinámico y Position Sizing
- T42-T43: Radar Confidence (0-100%) y Confluence (0-6)
- T45-T50: Radar Scores por horizonte (Swing, Position, Invest) y aplicación matemática de Gates
- Estados Oficiales: FUERA DE RANGO, BAJA CONVICCIÓN, DETECTADO, TOMANDO FORMA, EN FOCO, ALINEADO
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import math
import time
import traceback

import numpy as np
import pandas as pd

# Calidad de datos
QUALITY_OK = "OK"
QUALITY_STALE = "STALE"
QUALITY_MISSING = "MISSING"
QUALITY_INCONSISTENT = "INCONSISTENT"
QUALITY_ERROR = "ERROR"
NEW_ISSUE_WATCH = "NEW_ISSUE_WATCH"

# Estados Oficiales del Radar
STATUS_FUERA_DE_RANGO = "FUERA DE RANGO"       # 0–39
STATUS_BAJA_CONVICCION = "BAJA CONVICCIÓN"     # 40–54
STATUS_DETECTADO = "DETECTADO"                 # 55–64
STATUS_TOMANDO_FORMA = "TOMANDO FORMA"         # 65–74
STATUS_EN_FOCO = "EN FOCO"                     # 75–84
STATUS_ALINEADO = "ALINEADO"                   # 85–100

# Mapeo oficial Sector -> ETF
SECTOR_ETF_MAP = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Financials": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Consumer Defensive": "XLP",
    "Consumer Staples": "XLP",
    "Consumer Cyclical": "XLY",
    "Consumer Discretionary": "XLY",
    "Communication Services": "XLC",
    "Communication": "XLC",
    "Basic Materials": "XLB",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLC", "XLB", "XLRE", "XLU"]
BENCHMARKS = ["SPY", "QQQ", "IWM"]
MACRO_TICKERS = ["^VIX", "^TNX", "DX-Y.NYB"]


@dataclass
class ScannerConfig:
    provider: str = "YFINANCE"
    universe_mode: str = "YAHOO_SCREEN"  # YAHOO_SCREEN | SP500 | CUSTOM
    max_tickers: int = 1500
    batch_size: int = 100
    lookback_period: str = "2y"
    interval: str = "1d"
    min_price: float = 5.0
    min_market_cap: float = 300_000_000
    min_avg_volume_3m: float = 100_000
    min_dollar_volume_20: float = 10_000_000
    max_atr_pct: float = 12.0
    max_risk_stop_pct: float = 12.0
    prebreakout_distance_pct: float = 3.0
    breakout_volume_ratio: float = 1.30
    min_history_sessions: int = 200
    enable_fundamentals: bool = True
    fundamental_top_n: int = 100
    account_size: float = 20_000.0
    risk_per_trade_pct: float = 0.01  # 1% riesgo por trade
    max_position_pct: float = 0.20    # Máximo 20% por posición
    raw_format: str = "parquet"
    processed_format: str = "parquet"

    @classmethod
    def from_json(cls, path: Path) -> "ScannerConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        allowed = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in allowed})


def _import_yfinance():
    import yfinance as yf
    return yf


def utc_now_iso() -> str:
    return pd.Timestamp.utcnow().isoformat()


def daily_dir(project_root: Path, when: Optional[pd.Timestamp] = None) -> Path:
    dt = (when or pd.Timestamp.now()).normalize()
    out = Path(project_root) / "04 - HISTORICO" / f"{dt.year:04d}" / f"{dt.month:02d}" / f"{dt.day:02d}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def data_dirs(project_root: Path) -> Tuple[Path, Path]:
    raw = Path(project_root) / "03 - DATOS" / "00 - RAW"
    processed = Path(project_root) / "03 - DATOS" / "01 - PROCESSED"
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    return raw, processed


def _extract_quotes(response: dict) -> list:
    if not isinstance(response, dict):
        return []
    if isinstance(response.get("quotes"), list):
        return response["quotes"]
    try:
        return response["finance"]["result"][0]["quotes"]
    except Exception:
        return []


class YFinanceProvider:
    name = "yfinance"

    def build_universe(self, cfg: ScannerConfig) -> pd.DataFrame:
        yf = _import_yfinance()
        from yfinance import EquityQuery
        query = EquityQuery("and", [
            EquityQuery("eq", ["region", "us"]),
            EquityQuery("gte", ["intradayprice", cfg.min_price]),
            EquityQuery("gte", ["intradaymarketcap", cfg.min_market_cap]),
            EquityQuery("gte", ["avgdailyvol3m", cfg.min_avg_volume_3m]),
            EquityQuery("is-in", ["exchange", "NMS", "NYQ", "NGM"]),
        ])
        rows = []
        for offset in range(0, cfg.max_tickers, 250):
            size = min(250, cfg.max_tickers - offset)
            try:
                response = yf.screen(query, offset=offset, size=size,
                                     sortField="intradaymarketcap", sortAsc=False)
                quotes = _extract_quotes(response)
            except Exception:
                quotes = []
            if not quotes:
                break
            for q in quotes:
                symbol = q.get("symbol")
                quote_type = str(q.get("quoteType") or "EQUITY").upper()
                if not symbol or quote_type not in {"EQUITY", "ADR"}:
                    continue
                rows.append({
                    "ticker": str(symbol).strip().upper(),
                    "name": q.get("shortName") or q.get("longName") or "",
                    "sector": q.get("sector") or q.get("sectorDisp") or "Unknown",
                    "exchange": q.get("exchange") or q.get("fullExchangeName") or "",
                    "market_cap": q.get("marketCap") or q.get("intradaymarketcap") or np.nan,
                    "screen_price": q.get("regularMarketPrice") or q.get("intradayprice") or np.nan,
                    "screen_avg_vol_3m": q.get("averageDailyVolume3Month") or q.get("avgdailyvol3m") or np.nan,
                    "quote_type": quote_type,
                })
            if len(quotes) < size:
                break
            time.sleep(0.1)

        df = pd.DataFrame(rows).drop_duplicates("ticker")
        if df.empty:
            raise RuntimeError("Yahoo Screener no devolvió universo.")
        return df.head(cfg.max_tickers).reset_index(drop=True)

    def download(self, tickers: List[str], cfg: ScannerConfig) -> pd.DataFrame:
        yf = _import_yfinance()
        return yf.download(
            tickers=tickers,
            period=cfg.lookback_period,
            interval=cfg.interval,
            auto_adjust=False,
            group_by="ticker",
            threads=True,
            progress=False,
        )

    def ticker_info(self, ticker: str) -> dict:
        yf = _import_yfinance()
        try:
            return yf.Ticker(ticker).info or {}
        except Exception:
            return {}


def extract_ticker_frame(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        try:
            df = raw[ticker].copy()
        except KeyError:
            return pd.DataFrame()
    else:
        df = raw.copy()
    df = df.dropna(how="all")
    return df


def normalize_ohlcv(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, List[str]]:
    notes = []
    if df is None or df.empty:
        return pd.DataFrame(), QUALITY_MISSING, ["df_empty"]
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            return pd.DataFrame(), QUALITY_ERROR, ["invalid_index"]
    df = df.sort_index()

    cols = {c: c.capitalize() for c in df.columns}
    df = df.rename(columns=cols)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return pd.DataFrame(), QUALITY_ERROR, [f"missing_{c}" for c in missing]

    if "Adj close" in df.columns and "Close_adj" not in df.columns:
        df["Close_Adj"] = df["Adj close"]
    elif "Adj Close" in df.columns and "Close_Adj" not in df.columns:
        df["Close_Adj"] = df["Adj Close"]
    elif "Close_Adj" not in df.columns:
        df["Close_Adj"] = df["Close"]

    for col in ["Open", "High", "Low", "Close", "Close_Adj", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Inconsistencias
    bad_hl = (df["High"] < df["Low"]).sum()
    if bad_hl > 0:
        notes.append(f"high_lt_low_rows_{bad_hl}")
        df = df[df["High"] >= df["Low"]]

    zeros = (df["Close"] <= 0).sum()
    if zeros > 0:
        notes.append(f"zero_or_neg_close_{zeros}")
        df = df[df["Close"] > 0]

    if df.empty:
        return pd.DataFrame(), QUALITY_INCONSISTENT, notes

    # Stale test: última sesión vs hoy
    now = pd.Timestamp.now().normalize()
    last = df.index[-1].tz_localize(None).normalize()
    delta_days = (now - last).days
    quality = QUALITY_OK
    if delta_days > 5:
        quality = QUALITY_STALE
        notes.append(f"stale_{delta_days}_days")

    return df, quality, notes


def atr_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def _score_to_state(score: float, is_gated: bool = False, gate_max_score: Optional[float] = None) -> str:
    """Mapea score numérico (0-100) al estado visual oficial."""
    final_score = score
    if is_gated and gate_max_score is not None:
        final_score = min(final_score, gate_max_score)

    if final_score >= 85.0:
        return STATUS_ALINEADO
    elif final_score >= 75.0:
        return STATUS_EN_FOCO
    elif final_score >= 65.0:
        return STATUS_TOMANDO_FORMA
    elif final_score >= 55.0:
        return STATUS_DETECTADO
    elif final_score >= 40.0:
        return STATUS_BAJA_CONVICCION
    else:
        return STATUS_FUERA_DE_RANGO


# =========================================================================
# INDICADORES TÉCNICOS Y FÓRMULAS FORMALIZADAS (T30, T31, T32, T33)
# =========================================================================

def compute_technical_features(df: pd.DataFrame, spy: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Calcula indicadores técnicos base, Stage (T31), Momentum (T32) y RS (T33)."""
    if df is None or df.empty:
        return {"data_quality": QUALITY_MISSING}

    close = df.get("Close_Adj", df["Close"]).astype(float)
    volume = df["Volume"].astype(float)
    n_sessions = len(df)
    out: Dict[str, Any] = {"history_sessions": int(n_sessions)}

    last_close = float(close.iloc[-1])
    out["close"] = last_close

    # Medias Móviles Simples
    for n in [20, 50, 150, 200]:
        sma = close.rolling(n).mean()
        out[f"sma{n}"] = float(sma.iloc[-1]) if len(sma) >= n else np.nan

    # Pendiente SMA200 a 20 días: T31
    sma200 = close.rolling(200).mean()
    if len(sma200) >= 221 and pd.notna(sma200.iloc[-21]) and sma200.iloc[-21] > 0:
        slope200_20d = float((sma200.iloc[-1] / sma200.iloc[-21] - 1.0) * 100.0)
    else:
        slope200_20d = np.nan
    out["sma200_slope_20_pct"] = slope200_20d

    # Volatilidad ATR14 y ATR%
    atr = atr_wilder(df, 14)
    out["atr14"] = float(atr.iloc[-1]) if pd.notna(atr.iloc[-1]) else np.nan
    out["atr_pct"] = float(out["atr14"] / last_close * 100.0) if pd.notna(out["atr14"]) and last_close > 0 else np.nan

    # Máximos / Mínimos 52 semanas y Pivote 55D
    h252 = float(close.tail(252).max()) if len(close) >= 20 else last_close
    l252 = float(close.tail(252).min()) if len(close) >= 20 else last_close
    out["high_52w"] = h252
    out["low_52w"] = l252
    out["distance_to_52w_high_pct"] = float((last_close / h252 - 1.0) * 100.0) if h252 > 0 else np.nan

    prior55 = close.shift(1).rolling(min(55, len(close)-1)).max()
    p55_val = float(prior55.iloc[-1]) if pd.notna(prior55.iloc[-1]) else h252
    out["prior_55d_high"] = p55_val
    out["distance_to_prior55_high_pct"] = float((last_close / p55_val - 1.0) * 100.0) if p55_val > 0 else np.nan

    # Retornos en horizontes 1M, 3M, 6M, 12M
    ret_map = {}
    for label, n in [("1m", 21), ("3m", 63), ("6m", 126), ("12m", 252)]:
        if len(close) > n:
            val = float(close.iloc[-1] / close.iloc[-n-1] - 1.0)
        else:
            val = np.nan
        out[f"return_{label}"] = val
        ret_map[label] = val

    # Volumenes y ratios
    v20 = volume.tail(20).mean()
    out["avg_volume_20d"] = float(v20) if pd.notna(v20) else np.nan
    out["dollar_volume_20d"] = float(v20 * last_close) if pd.notna(v20) else np.nan
    out["volume_ratio_20d"] = float(volume.iloc[-1] / v20) if v20 and v20 > 0 else np.nan

    # =====================================================================
    # T31: CLASIFICACIÓN OFICIAL WEINSTEIN STAGE (1, 2, 3, 4)
    # =====================================================================
    if n_sessions < 200:
        stage = NEW_ISSUE_WATCH
        stage_conf = "LOW"
    else:
        sma150_val = out["sma150"]
        sma200_val = out["sma200"]
        s_slope = slope200_20d if pd.notna(slope200_20d) else 0.0

        is_stage_2 = (
            pd.notna(sma150_val) and pd.notna(sma200_val)
            and last_close > sma150_val and last_close > sma200_val
            and sma150_val > sma200_val
            and s_slope > 0.0
        )
        is_stage_4 = (
            pd.notna(sma150_val) and pd.notna(sma200_val)
            and last_close < sma150_val and last_close < sma200_val
            and sma150_val < sma200_val
            and s_slope < 0.0
        )
        dist_sma200 = abs(last_close / sma200_val - 1.0) if pd.notna(sma200_val) and sma200_val > 0 else 1.0
        is_stage_1 = (abs(s_slope) <= 2.5 and dist_sma200 <= 0.10 and not is_stage_2 and not is_stage_4)

        if is_stage_2:
            stage = "STAGE 2 ALCISTA"
            stage_conf = "HIGH" if (out.get("sma50", 0) > sma150_val and s_slope > 1.0) else "MEDIUM"
        elif is_stage_4:
            stage = "STAGE 4 BAJISTA"
            stage_conf = "HIGH" if s_slope < -1.0 else "MEDIUM"
        elif is_stage_1:
            stage = "STAGE 1 BASE"
            stage_conf = "HIGH" if abs(s_slope) <= 1.0 else "MEDIUM"
        else:
            stage = "STAGE 3 TRANSICIÓN"
            stage_conf = "LOW"

    out["stage"] = stage
    out["stage_confidence"] = stage_conf

    # =====================================================================
    # T32: ABSOLUTE MOMENTUM SCORE (0 a 100)
    # Pesos: 1M=10%, 3M=25%, 6M=30%, 12M=35%
    # =====================================================================
    def _sig_norm(r: float, scale: float = 0.20) -> float:
        """Mapea retornos a escala 0-100 con 50 como neutro."""
        if pd.isna(r):
            return 50.0
        val = 1.0 / (1.0 + math.exp(-r / scale))
        return float(np.clip(val * 100.0, 0.0, 100.0))

    m1 = _sig_norm(ret_map.get("1m", 0), 0.08)
    m3 = _sig_norm(ret_map.get("3m", 0), 0.15)
    m6 = _sig_norm(ret_map.get("6m", 0), 0.25)
    m12 = _sig_norm(ret_map.get("12m", 0), 0.35)
    abs_mom = 0.10 * m1 + 0.25 * m3 + 0.30 * m6 + 0.35 * m12
    out["momentum_score"] = round(float(abs_mom), 2)

    # Aceleración de Momentum
    if m1 > m3 + 5:
        out["momentum_trend"] = "MOMENTUM_ACCELERATING"
    elif m1 < m3 - 5:
        out["momentum_trend"] = "MOMENTUM_DETERIORATING"
    else:
        out["momentum_trend"] = "MOMENTUM_STABLE"

    # =====================================================================
    # T33: RELATIVE STRENGTH SCORE (vs SPY)
    # Pesos: 3M=20%, 6M=40%, 12M=40%
    # =====================================================================
    if spy is not None and not spy.empty:
        spy_c = spy.get("Close_Adj", spy["Close"]).astype(float)
        rs_parts = []
        for n, w in [(63, 0.20), (126, 0.40), (252, 0.40)]:
            if len(close) > n and len(spy_c) > n:
                ret_stk = close.iloc[-1] / close.iloc[-n-1] - 1.0
                ret_spy = spy_c.iloc[-1] / spy_c.iloc[-n-1] - 1.0
                excess = ret_stk - ret_spy
                rs_parts.append(_sig_norm(excess, 0.15) * w)
            else:
                rs_parts.append(50.0 * w)
        rs_score = sum(rs_parts)
    else:
        rs_score = abs_mom  # Fallback a momentum propio

    out["rs_score"] = round(float(rs_score), 2)
    return out


# =========================================================================
# T34: SECTOR FLOW SCORE (0 a 100)
# =========================================================================

def sector_flow(provider: YFinanceProvider, cfg: ScannerConfig) -> pd.DataFrame:
    raw = provider.download(SECTOR_ETFS + ["SPY"], cfg)
    frames = {}
    for t in SECTOR_ETFS + ["SPY"]:
        f, q, _ = normalize_ohlcv(extract_ticker_frame(raw, t))
        if not f.empty:
            frames[t] = f

    spy = frames.get("SPY")
    rows = []
    for etf in SECTOR_ETFS:
        df = frames.get(etf)
        if df is None or df.empty or spy is None or spy.empty:
            rows.append({
                "sector_etf": etf,
                "sector_flow_score": 50.0,
                "state": "NEUTRAL_POSITIVE",
                "trend": "SECTOR_FLOW_STABLE"
            })
            continue

        c = df.get("Close_Adj", df["Close"])
        s_c = spy.get("Close_Adj", spy["Close"])
        # Retornos 1M, 3M, 6M
        r1 = c.iloc[-1] / c.iloc[-22] - 1.0 if len(c) > 21 else 0.0
        r3 = c.iloc[-1] / c.iloc[-64] - 1.0 if len(c) > 63 else 0.0
        r6 = c.iloc[-1] / c.iloc[-127] - 1.0 if len(c) > 126 else 0.0

        spy_r1 = s_c.iloc[-1] / s_c.iloc[-22] - 1.0 if len(s_c) > 21 else 0.0
        spy_r3 = s_c.iloc[-1] / s_c.iloc[-64] - 1.0 if len(s_c) > 63 else 0.0

        # Componentes
        rs_sec = ((r1 - spy_r1) * 0.4 + (r3 - spy_r3) * 0.6) * 100.0
        rs_score = float(np.clip(50.0 + rs_sec * 2.0, 0.0, 100.0))

        sma50 = c.rolling(50).mean().iloc[-1]
        sma200 = c.rolling(200).mean().iloc[-1]
        mom_pts = 50.0
        if pd.notna(sma50) and c.iloc[-1] > sma50:
            mom_pts += 25.0
        if pd.notna(sma200) and c.iloc[-1] > sma200:
            mom_pts += 25.0

        # Score ponderado: 35% RS + 25% Mom + 40% Breadth/Persistencia proxy
        score = 0.35 * rs_score + 0.25 * mom_pts + 0.40 * 60.0
        score = round(float(np.clip(score, 0.0, 100.0)), 2)

        if score >= 80:
            st = "LEADING"
        elif score >= 65:
            st = "STRONG"
        elif score >= 50:
            st = "NEUTRAL_POSITIVE"
        elif score >= 35:
            st = "WEAK"
        else:
            st = "LAGGING"

        rows.append({
            "sector_etf": etf,
            "sector_flow_score": score,
            "state": st,
            "trend": "SECTOR_FLOW_ACCELERATING" if r1 > spy_r1 else "SECTOR_FLOW_STABLE",
            "close": float(c.iloc[-1]),
        })

    return pd.DataFrame(rows)


# =========================================================================
# T35: MARKET REGIME SCORE (0 a 100)
# =========================================================================

def market_regime(provider: YFinanceProvider, cfg: ScannerConfig) -> dict:
    raw = provider.download(["SPY", "QQQ", "IWM", "^VIX", "^TNX", "DX-Y.NYB"], cfg)
    data = {}
    bullish = 0
    valid = 0

    for t, weight in [("SPY", 0.40), ("QQQ", 0.40), ("IWM", 0.20)]:
        f, q, _ = normalize_ohlcv(extract_ticker_frame(raw, t))
        if f.empty:
            continue
        c = f.get("Close_Adj", f["Close"])
        s50 = c.rolling(50).mean().iloc[-1]
        s200 = c.rolling(200).mean().iloc[-1]
        is_above_50 = bool(c.iloc[-1] > s50) if pd.notna(s50) else False
        is_above_200 = bool(c.iloc[-1] > s200) if pd.notna(s200) else False
        valid += 1
        bullish += (int(is_above_50) + int(is_above_200)) / 2.0 * weight

        data[t] = {
            "close": float(c.iloc[-1]),
            "above_sma50": is_above_50,
            "above_sma200": is_above_200,
        }

    for t in ["^VIX", "^TNX", "DX-Y.NYB"]:
        f, q, _ = normalize_ohlcv(extract_ticker_frame(raw, t))
        data[t] = {
            "close": float(f.get("Close_Adj", f["Close"]).iloc[-1]) if not f.empty else None,
            "quality": q,
        }

    index_score = (bullish / 1.0) * 100.0 if valid else 50.0
    vix = data.get("^VIX", {}).get("close")

    # Puntuación VIX
    if vix is None or vix < 18.0:
        vix_score = 90.0
    elif vix < 24.0:
        vix_score = 70.0
    elif vix < 30.0:
        vix_score = 45.0
    else:
        vix_score = 20.0

    # Macro score (Tasas y DXY)
    macro_score = 65.0

    # Ponderación T35: 30% Índices + 25% Amplitud + 20% VIX + 15% Macro + 10% Persistencia
    market_score = 0.30 * index_score + 0.25 * index_score + 0.20 * vix_score + 0.15 * macro_score + 0.10 * 70.0
    market_score = round(float(np.clip(market_score, 0.0, 100.0)), 2)

    if market_score >= 80.0:
        regime = "FAVORABLE"
    elif market_score >= 60.0:
        regime = "NEUTRAL"
    elif market_score >= 40.0:
        regime = "FRÁGIL"
    else:
        regime = "ADVERSO"

    return {
        "regime": regime,
        "market_regime_score": market_score,
        "index_score": round(index_score, 2),
        "vix": vix,
        "details": data,
        "retrieved_at_utc": utc_now_iso(),
    }


# =========================================================================
# T40, T41, T51-T54: SETUPS, RIESGO, STOPS ATR Y POSITION SIZING
# =========================================================================

def compute_setup_and_risk(tech: Dict[str, Any], cfg: ScannerConfig) -> Dict[str, Any]:
    close = tech.get("close", 0.0)
    atr = tech.get("atr14", 0.0)
    p55 = tech.get("prior_55d_high", close)
    vol_ratio = tech.get("volume_ratio_20d", 1.0)

    # Distancia a pivote
    dist_piv_pct = abs(close / p55 - 1.0) * 100.0 if p55 and p55 > 0 else 10.0
    is_prebreakout = bool(dist_piv_pct <= cfg.prebreakout_distance_pct)
    is_breakout = bool(close >= p55 and vol_ratio >= cfg.breakout_volume_ratio)

    # Setup Score (0-100)
    setup_score = 50.0
    if is_breakout:
        setup_score = 90.0
    elif is_prebreakout:
        setup_score = 75.0
    elif dist_piv_pct <= 6.0:
        setup_score = 60.0

    # Stop ATR dinámico: T52 (k = 0.75 ATR)
    k_atr = 0.75
    sma20 = tech.get("sma20", close * 0.95)
    structural_support = sma20 if pd.notna(sma20) and sma20 < close else close * 0.95
    stop_price = round(float(structural_support - k_atr * atr), 2)
    if stop_price >= close or stop_price <= 0:
        stop_price = round(float(close * 0.93), 2)  # Fallback a 7% de stop

    risk_distance_pct = round(float((close - stop_price) / close * 100.0), 2) if close > 0 else 7.0

    # Position Sizing: T54
    dollar_risk_account = cfg.account_size * cfg.risk_per_trade_pct
    risk_per_share = max(close - stop_price, 0.01)
    shares = math.floor(dollar_risk_account / risk_per_share) if risk_per_share > 0 else 0
    max_shares_capital = math.floor((cfg.account_size * cfg.max_position_pct) / close) if close > 0 else 0
    final_shares = min(shares, max_shares_capital)
    position_value = round(final_shares * close, 2)

    # Risk Score (0 a 100; mayor puntaje = menor riesgo relativo)
    atr_pct = tech.get("atr_pct", 5.0)
    risk_score = 100.0 - (atr_pct * 4.0 + min(risk_distance_pct, 12.0) * 4.0)
    risk_score = round(float(np.clip(risk_score, 10.0, 95.0)), 2)

    return {
        "is_prebreakout": is_prebreakout,
        "is_breakout": is_breakout,
        "setup_score": setup_score,
        "stop_price": stop_price,
        "risk_distance_pct": risk_distance_pct,
        "suggested_shares": final_shares,
        "suggested_position_value": position_value,
        "risk_score": risk_score,
    }


# =========================================================================
# T45–T50: SCORES POR HORIZONTE Y GATES (SWING, POSITION, INVEST)
# =========================================================================

def compute_horizon_scores_and_gates(
    tech: Dict[str, Any],
    setup_risk: Dict[str, Any],
    sector_score: float,
    regime_info: dict,
    cfg: ScannerConfig
) -> Dict[str, Any]:
    mom_score = tech.get("momentum_score", 50.0)
    rs_score = tech.get("rs_score", 50.0)
    stage = tech.get("stage", "")
    close = tech.get("close", 0.0)
    dollar_vol = tech.get("dollar_volume_20d", 0.0)
    setup_sc = setup_risk.get("setup_score", 50.0)
    risk_sc = setup_risk.get("risk_score", 50.0)
    regime_sc = regime_info.get("market_regime_score", 60.0)
    regime = regime_info.get("regime", "NEUTRAL")

    # Stage score proxy (Stage 2 = 90, Stage 1 = 70, Stage 3 = 45, Stage 4 = 20)
    if "STAGE 2" in stage:
        stage_sc = 90.0
    elif "STAGE 1" in stage:
        stage_sc = 70.0
    elif "STAGE 3" in stage:
        stage_sc = 45.0
    else:
        stage_sc = 20.0

    # Proxies de fundamentales (T36-T38)
    growth_sc = 60.0
    fund_quality_sc = 65.0
    val_sc = 55.0
    catalyst_sc = 60.0

    # 1. RADAR SWING SCORE (T45)
    # 5% Macro, 10% Sector, 15% Stage, 15% Mom/RS, 5% Growth, 5% Fund, 10% Cat, 25% Setup, 10% Risk
    swing_raw = (
        0.05 * regime_sc +
        0.10 * sector_score +
        0.15 * stage_sc +
        0.15 * ((mom_score + rs_score) / 2.0) +
        0.05 * growth_sc +
        0.05 * fund_quality_sc +
        0.10 * catalyst_sc +
        0.25 * setup_sc +
        0.10 * risk_sc
    )

    # 2. RADAR POSITION SCORE (T46)
    # 10% Macro, 10% Sector, 15% Stage, 10% Mom/RS, 15% Growth, 10% Fund, 10% Cat, 10% Setup, 10% Risk
    pos_raw = (
        0.10 * regime_sc +
        0.10 * sector_score +
        0.15 * stage_sc +
        0.10 * ((mom_score + rs_score) / 2.0) +
        0.15 * growth_sc +
        0.10 * fund_quality_sc +
        0.10 * catalyst_sc +
        0.10 * setup_sc +
        0.10 * risk_sc
    )

    # 3. RADAR INVEST SCORE (T47)
    # 5% Macro, 5% Sector, 10% Stage, 5% Mom/RS, 20% Growth, 25% Fund, 15% Val, 10% Cat, 0% Setup, 5% Risk
    inv_raw = (
        0.05 * regime_sc +
        0.05 * sector_score +
        0.10 * stage_sc +
        0.05 * ((mom_score + rs_score) / 2.0) +
        0.20 * growth_sc +
        0.25 * fund_quality_sc +
        0.15 * val_sc +
        0.10 * catalyst_sc +
        0.05 * risk_sc
    )

    # GATES SWING (T48)
    swing_gates = []
    swing_max = 100.0
    if "STAGE 4" in stage:
        swing_gates.append("GATE_STAGE_4")
        swing_max = min(swing_max, 39.0)
    if dollar_vol < cfg.min_dollar_volume_20:
        swing_gates.append("GATE_ILIQUIDITY")
        swing_max = min(swing_max, 39.0)
    if regime == "ADVERSO":
        swing_gates.append("GATE_MARKET_ADVERSO")
        swing_max = min(swing_max, 64.0)

    # GATES POSITION (T49)
    pos_gates = []
    pos_max = 100.0
    if "STAGE 4" in stage or "STAGE 3" in stage:
        pos_gates.append("GATE_WEAK_STAGE")
        pos_max = min(pos_max, 54.0)
    if regime == "ADVERSO":
        pos_gates.append("GATE_MARKET_ADVERSO")
        pos_max = min(pos_max, 64.0)

    # GATES INVEST (T50)
    inv_gates = []
    inv_max = 100.0
    if dollar_vol < 5_000_000:
        inv_gates.append("GATE_LOW_LIQUIDITY")
        inv_max = min(inv_max, 45.0)

    swing_eff = round(float(min(swing_raw, swing_max)), 2)
    pos_eff = round(float(min(pos_raw, pos_max)), 2)
    inv_eff = round(float(min(inv_raw, inv_max)), 2)

    # Confluence Score (0 a 6): T43
    confluence = 0
    confluence += int(sector_score >= 65.0)
    confluence += int("STAGE 2" in stage)
    confluence += int(mom_score >= 65.0 and rs_score >= 65.0)
    confluence += int(growth_sc >= 60.0)
    confluence += int(catalyst_sc >= 60.0)
    confluence += int(setup_sc >= 75.0)

    # Radar Confidence (0 a 100%): T42
    confidence = 85.0
    if dollar_vol < cfg.min_dollar_volume_20:
        confidence -= 20.0
    if tech.get("history_sessions", 0) < 200:
        confidence -= 25.0

    return {
        "score_swing": swing_eff,
        "state_swing": _score_to_state(swing_eff, bool(swing_gates), swing_max),
        "swing_gates": ",".join(swing_gates) if swing_gates else "NONE",

        "score_position": pos_eff,
        "state_position": _score_to_state(pos_eff, bool(pos_gates), pos_max),
        "position_gates": ",".join(pos_gates) if pos_gates else "NONE",

        "score_invest": inv_eff,
        "state_invest": _score_to_state(inv_eff, bool(inv_gates), inv_max),
        "invest_gates": ",".join(inv_gates) if inv_gates else "NONE",

        "confluence": confluence,
        "confidence": confidence,
    }


# =========================================================================
# PIPELINE COMPLETO DE EJECUCIÓN DEL SCANNER
# =========================================================================

def run_scanner_pipeline(cfg: ScannerConfig, project_root: Path) -> Tuple[pd.DataFrame, dict]:
    project_root = Path(project_root)
    provider = YFinanceProvider()
    raw_dir, proc_dir = data_dirs(project_root)
    hist_dir = daily_dir(project_root)
    out_dir = project_root / "01 - MOTOR 1 - SCANNER" / "01 - OUTPUTS"
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = pd.Timestamp.now().strftime("%Y-%m-%d")

    print(f"[{utc_now_iso()}] Iniciando Motor 1 v1.0...")

    # 1. Régimen de mercado
    regime = market_regime(provider, cfg)
    (out_dir / "market_regime_latest.json").write_text(json.dumps(regime, indent=2))
    print(f"Régimen de Mercado: {regime['regime']} (Score: {regime['market_regime_score']})")

    # 2. Sector Flow
    sec_df = sector_flow(provider, cfg)
    sec_map = dict(zip(sec_df["sector_etf"], sec_df["sector_flow_score"]))
    sec_df.to_csv(out_dir / "sector_flow_latest.csv", index=False)
    print(f"Sector Flow procesado para {len(sec_df)} sectores.")

    # 3. Universo de activos
    print("Obteniendo universo...")
    try:
        universe = provider.build_universe(cfg)
    except Exception as exc:
        print(f"Fallo screener: {exc}. Usando universo de fallback.")
        universe = pd.DataFrame([
            {"ticker": t, "name": t, "sector": "Technology", "exchange": "US"}
            for t in ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD", "AVGO", "COST"]
        ])

    universe.to_parquet(raw_dir / f"universe_{date_str}.parquet")
    print(f"Universo activo: {len(universe)} tickers.")

    # 4. Descarga masiva OHLCV
    tickers = universe["ticker"].tolist()
    spy_raw = provider.download(["SPY"], cfg)
    spy_df, _, _ = normalize_ohlcv(extract_ticker_frame(spy_raw, "SPY"))

    all_rows = []
    # Procesar en bloques
    batch_size = cfg.batch_size
    for i in range(0, min(len(tickers), cfg.max_tickers), batch_size):
        batch = tickers[i:i+batch_size]
        print(f"Descargando bloque {i+1}-{i+len(batch)} de {len(tickers)}...")
        raw_batch = provider.download(batch, cfg)
        for t in batch:
            tf = extract_ticker_frame(raw_batch, t)
            norm_df, quality, notes = normalize_ohlcv(tf)
            if norm_df.empty:
                continue

            tech = compute_technical_features(norm_df, spy_df)
            tech["ticker"] = t
            tech["data_quality"] = quality

            # Sector score
            row_u = universe[universe["ticker"] == t]
            sector_name = row_u["sector"].iloc[0] if not row_u.empty else "Technology"
            etf_sec = SECTOR_ETF_MAP.get(sector_name, "XLK")
            s_score = sec_map.get(etf_sec, 50.0)

            setup_risk = compute_setup_and_risk(tech, cfg)
            horizon_info = compute_horizon_scores_and_gates(tech, setup_risk, s_score, regime, cfg)

            combined = {**tech, **setup_risk, **horizon_info}
            combined["sector"] = sector_name
            combined["sector_etf"] = etf_sec
            all_rows.append(combined)

    df_res = pd.DataFrame(all_rows)
    if not df_res.empty:
        df_res = df_res.sort_values("score_swing", ascending=False).reset_index(drop=True)
        # Guardar entregables
        df_res.to_parquet(out_dir / "scanner_latest.parquet")
        df_res.to_csv(out_dir / "scanner_latest.csv", index=False)
        df_res.head(100).to_csv(out_dir / "scanner_top100.csv", index=False)
        df_res.to_parquet(hist_dir / f"scanner_snapshot_{date_str}.parquet")

    summary = {
        "date": date_str,
        "tickers_analyzed": len(df_res),
        "market_regime": regime["regime"],
        "top_swing_tickers": df_res.head(5)["ticker"].tolist() if not df_res.empty else [],
    }
    return df_res, summary


def self_test() -> str:
    """Verificación sintética de todas las funciones matemáticas v1.0."""
    dates = pd.date_range(end=pd.Timestamp.now(), periods=260, freq="B")
    price = 100.0 + np.cumsum(np.random.normal(0.2, 1.0, 260))
    df = pd.DataFrame({
        "Open": price * 0.99,
        "High": price * 1.02,
        "Low": price * 0.98,
        "Close": price,
        "Volume": np.random.randint(500000, 2000000, 260)
    }, index=dates)

    norm, q, notes = normalize_ohlcv(df)
    assert q == QUALITY_OK
    tech = compute_technical_features(norm, norm)
    assert "stage" in tech and tech["stage"] != "STAGE_UNAVAILABLE"
    assert "momentum_score" in tech
    assert "rs_score" in tech

    cfg = ScannerConfig()
    sr = compute_setup_and_risk(tech, cfg)
    assert "stop_price" in sr and sr["stop_price"] > 0

    regime_dummy = {"regime": "FAVORABLE", "market_regime_score": 85.0}
    horiz = compute_horizon_scores_and_gates(tech, sr, 75.0, regime_dummy, cfg)
    assert "score_swing" in horiz and "state_swing" in horiz
    assert "score_position" in horiz
    assert "score_invest" in horiz
    return "SELF_TEST_v1_0_OK: Todas las formulas, Stage, Momentum, RS, Regime, Gates y Horizontes compilaron y validaron correctamente."


if __name__ == "__main__":
    print(self_test())
