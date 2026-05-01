# coding=utf-8
"""
Real-time voice EQ for the streaming TTS pipeline.

Two biquad sections in series:
  1. Low-shelf at f_ls / gain_ls_db   (default 100 Hz, +3 dB) — adds bass body
  2. Peaking bell at f_b / gain_b_db / Q_b  (default 200 Hz, +2 dB, Q=1.0)
                                         — adds low-mid warmth without muddying
                                           the 300–500 Hz "boxiness" zone

Both are second-order IIR filters from the RBJ Audio EQ Cookbook. Per-request
state (`zi`) is carried across chunks so there are no audible clicks at
chunk boundaries.

Defaults are conservative and off. Enable + tune via env vars at server
startup:

  TTS_EQ_ENABLE=1                  # required to activate
  TTS_EQ_LOW_SHELF_HZ=100          # low-shelf corner frequency (Hz)
  TTS_EQ_LOW_SHELF_DB=3.0          # low-shelf gain (dB; positive = boost)
  TTS_EQ_BELL_HZ=200               # peaking bell center (Hz)
  TTS_EQ_BELL_DB=2.0               # peaking bell gain (dB)
  TTS_EQ_BELL_Q=1.0                # peaking bell Q (sharpness; ~0.7-2.0 sane range)

If you want only one of the two stages, set the other's dB to 0 — it then
collapses to a unity filter and is effectively a no-op.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass

import numpy as np
from scipy.signal import sosfilt, sosfilt_zi

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EQConfig:
    enabled: bool
    sample_rate: int
    low_shelf_hz: float
    low_shelf_db: float
    bell_hz: float
    bell_db: float
    bell_q: float

    @classmethod
    def from_env(cls, sample_rate: int) -> "EQConfig":
        def _f(name: str, default: float) -> float:
            v = os.environ.get(name)
            return float(v) if v is not None else default

        return cls(
            enabled=os.environ.get("TTS_EQ_ENABLE", "0").lower() in ("1", "true", "yes"),
            sample_rate=sample_rate,
            low_shelf_hz=_f("TTS_EQ_LOW_SHELF_HZ", 100.0),
            low_shelf_db=_f("TTS_EQ_LOW_SHELF_DB", 3.0),
            bell_hz=_f("TTS_EQ_BELL_HZ", 200.0),
            bell_db=_f("TTS_EQ_BELL_DB", 2.0),
            bell_q=_f("TTS_EQ_BELL_Q", 1.0),
        )


def _low_shelf_biquad(f0: float, gain_db: float, fs: int,
                      slope: float = 1.0) -> np.ndarray:
    """RBJ low-shelf. Returns [b0,b1,b2,a0,a1,a2] (a0 normalized to 1).
    slope=1.0 corresponds to Butterworth-like shelving (no overshoot).
    """
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / fs
    cos_w = math.cos(w0)
    sin_w = math.sin(w0)
    # Use slope (S) parameterization from RBJ cookbook
    alpha = (sin_w / 2.0) * math.sqrt((A + 1.0 / A) * (1.0 / slope - 1.0) + 2.0)
    sqrtA_alpha = 2.0 * math.sqrt(A) * alpha

    b0 = A * ((A + 1) - (A - 1) * cos_w + sqrtA_alpha)
    b1 = 2 * A * ((A - 1) - (A + 1) * cos_w)
    b2 = A * ((A + 1) - (A - 1) * cos_w - sqrtA_alpha)
    a0 = (A + 1) + (A - 1) * cos_w + sqrtA_alpha
    a1 = -2 * ((A - 1) + (A + 1) * cos_w)
    a2 = (A + 1) + (A - 1) * cos_w - sqrtA_alpha
    return np.array([b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0],
                    dtype=np.float64)


def _peaking_biquad(f0: float, gain_db: float, Q: float, fs: int) -> np.ndarray:
    """RBJ peaking EQ (bell). Returns [b0,b1,b2,a0,a1,a2] normalized."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / fs
    cos_w = math.cos(w0)
    sin_w = math.sin(w0)
    alpha = sin_w / (2.0 * Q)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w
    a2 = 1.0 - alpha / A
    return np.array([b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0],
                    dtype=np.float64)


class VoiceEQ:
    """Cascade of low-shelf + peaking biquad. Stateful per-request."""

    def __init__(self, cfg: EQConfig):
        self.cfg = cfg
        if not cfg.enabled:
            self.sos: np.ndarray | None = None
            return

        sections = []
        if abs(cfg.low_shelf_db) > 1e-3:
            ls = _low_shelf_biquad(cfg.low_shelf_hz, cfg.low_shelf_db,
                                  cfg.sample_rate)
            sections.append(ls[:6])
        if abs(cfg.bell_db) > 1e-3:
            bp = _peaking_biquad(cfg.bell_hz, cfg.bell_db, cfg.bell_q,
                                cfg.sample_rate)
            sections.append(bp[:6])

        if not sections:
            self.sos = None
            return

        # sosfilt expects shape (n_sections, 6) with [b0,b1,b2,a0,a1,a2].
        self.sos = np.vstack(sections).astype(np.float64)
        log.info(
            "VoiceEQ enabled: low_shelf=%.1fHz/%+.1fdB  bell=%.1fHz/%+.1fdB Q=%.2f  "
            "(sections=%d)",
            cfg.low_shelf_hz, cfg.low_shelf_db,
            cfg.bell_hz, cfg.bell_db, cfg.bell_q,
            len(sections),
        )

    @property
    def active(self) -> bool:
        return self.sos is not None

    def make_state(self) -> np.ndarray | None:
        """Per-request filter memory. Pass into process() for the first chunk;
        process() returns the updated state to pass into the next chunk.
        Returns None if EQ is inactive (caller should bypass process()).
        """
        if self.sos is None:
            return None
        # sosfilt_zi gives the steady-state response to unit input; for clean
        # silence-start this is OK because audio starts near 0 anyway.
        return sosfilt_zi(self.sos).astype(np.float64)

    def process(self, x: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Filter one chunk. x is float32/float64 mono in [-1, 1].

        Returns (filtered, new_state). new_state must be passed to the next
        process() call for this request to avoid clicks at chunk boundaries.
        """
        if self.sos is None:
            return x, state
        x64 = x.astype(np.float64, copy=False)
        y, new_state = sosfilt(self.sos, x64, zi=state)
        return y.astype(x.dtype, copy=False), new_state
