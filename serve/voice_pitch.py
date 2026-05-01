# coding=utf-8
"""
Real-time pitch shift for the streaming TTS pipeline.

Uses librosa's STFT phase vocoder under the hood. To stream cleanly across
2-second chunks (each emitted independently by vllm-omni), we prepend the
last N samples of the previous chunk to give the STFT proper context, then
discard the corresponding output prefix. This eliminates the audible click
at chunk boundaries you'd get from naively pitch-shifting each chunk.

Quality notes:
  - Phase vocoder pitch shift does NOT preserve formants. At +2 semitones
    the voice sounds higher AND slightly "younger" (formants shift with
    pitch). For a "girlier" male-voice flavor that's usually what you want.
  - At ±3 semitones quality is acceptable; ±5 starts to sound chipmunk-y.
    For larger shifts a formant-preserving algorithm (WORLD vocoder, PSOLA)
    is required.
  - There IS extra CPU per chunk — librosa.pitch_shift takes ~50-100 ms
    for a 2-s chunk on CPU. Adds to per-chunk wall time, but vllm-omni
    synth is faster than playback (RTF~0.35) so playback stays gapless.

Env config (off unless TTS_PITCH_SEMITONES is set non-zero):

  TTS_PITCH_SEMITONES=2.0            # positive = higher pitch
  TTS_PITCH_BIN_PER_OCTAVE=12        # bins per octave (12 = standard semitones)
  TTS_PITCH_OVERLAP_MS=20            # raw-audio context fed to STFT for continuity
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PitchConfig:
    semitones: float
    bins_per_octave: int
    overlap_samples: int

    @classmethod
    def from_env(cls, sample_rate: int) -> "PitchConfig":
        semis = float(os.environ.get("TTS_PITCH_SEMITONES", "0"))
        bpo = int(os.environ.get("TTS_PITCH_BINS_PER_OCTAVE", "12"))
        overlap_ms = float(os.environ.get("TTS_PITCH_OVERLAP_MS", "20"))
        overlap_samples = int(round(sample_rate * overlap_ms / 1000.0))
        return cls(semitones=semis, bins_per_octave=bpo,
                  overlap_samples=overlap_samples)


class PitchShifter:
    """Per-request stateful streaming pitch shifter."""

    def __init__(self, cfg: PitchConfig, sample_rate: int):
        self.cfg = cfg
        self.sr = sample_rate
        self.active = abs(cfg.semitones) > 1e-3
        self._librosa = None
        if self.active:
            try:
                import librosa  # type: ignore
                self._librosa = librosa
                log.info(
                    "PitchShifter enabled: %+.2f semitones (bpo=%d, overlap=%d samples)",
                    cfg.semitones, cfg.bins_per_octave, cfg.overlap_samples,
                )
            except ImportError:
                log.error(
                    "TTS_PITCH_SEMITONES set but librosa is not installed in this env. "
                    "Run: pip install librosa  (then restart). Pitch shift DISABLED."
                )
                self.active = False

    def make_state(self) -> dict | None:
        if not self.active:
            return None
        # prev_tail holds the last N RAW input samples to be re-fed as
        # STFT context on the next chunk. None for the first chunk.
        return {"prev_tail": np.zeros(0, dtype=np.float32)}

    def process(self, x: np.ndarray, state: dict) -> tuple[np.ndarray, dict]:
        """Pitch-shift one chunk. x is float mono in [-1, 1]. Returns
        (shifted, new_state). Output length matches input length.
        """
        if not self.active:
            return x, state
        prev_tail = state["prev_tail"]
        if prev_tail.size > 0:
            extended = np.concatenate([prev_tail, x.astype(np.float32, copy=False)])
            offset = prev_tail.size
        else:
            extended = x.astype(np.float32, copy=False)
            offset = 0

        shifted = self._librosa.effects.pitch_shift(
            extended,
            sr=self.sr,
            n_steps=self.cfg.semitones,
            bins_per_octave=self.cfg.bins_per_octave,
        )
        out = shifted[offset:]

        # Stash tail of THIS chunk's raw input for the next call.
        if x.size >= self.cfg.overlap_samples:
            state["prev_tail"] = x[-self.cfg.overlap_samples:].astype(np.float32, copy=True)
        else:
            state["prev_tail"] = x.astype(np.float32, copy=True)

        return out.astype(x.dtype, copy=False), state
