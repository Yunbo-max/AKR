from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt


@dataclass(frozen=True)
class Intervention:
    name: str
    low_hz: float | None = None
    high_hz: float | None = None


def _mono_float32(waveform: np.ndarray) -> np.ndarray:
    x = np.asarray(waveform, dtype=np.float32)
    if x.ndim == 2:
        # soundfile convention is [time, channel].
        x = x.mean(axis=1)
    if x.ndim != 1:
        raise ValueError(f"expected mono or [time, channel], got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("waveform contains NaN or infinity")
    return x


def _sos_filter(x: np.ndarray, sample_rate: int, kind: str, cutoff, order: int) -> np.ndarray:
    nyquist = sample_rate / 2
    wn = np.asarray(cutoff, dtype=float) / nyquist
    if np.any(wn <= 0) or np.any(wn >= 1):
        raise ValueError(f"cutoff {cutoff} must lie strictly inside (0, {nyquist})")
    sos = butter(order, wn, btype=kind, output="sos")
    # filtfilt avoids frequency-dependent delay. Very short events cannot support
    # the default padding, so pad explicitly and then crop.
    min_len = 6 * order + 1
    if len(x) < min_len:
        pad = min_len - len(x)
        xpad = np.pad(x, (pad // 2, pad - pad // 2), mode="reflect")
        y = sosfiltfilt(sos, xpad)
        return y[pad // 2 : pad // 2 + len(x)].astype(np.float32)
    return sosfiltfilt(sos, x).astype(np.float32)


def apply_intervention(
    waveform: np.ndarray,
    sample_rate: int,
    intervention: Intervention,
    order: int = 10,
) -> np.ndarray:
    """Filter at the original sample rate, before the model's 16 kHz resampling."""
    x = _mono_float32(waveform)
    if intervention.name == "full":
        return x.copy()
    if intervention.name == "lowpass":
        if intervention.high_hz is None:
            raise ValueError("lowpass requires high_hz")
        # If the original recording's Nyquist is already below the requested
        # cutoff, the signal is already band-limited and this is a true no-op.
        if intervention.high_hz >= sample_rate / 2:
            return x.copy()
        return _sos_filter(x, sample_rate, "lowpass", intervention.high_hz, order)
    if intervention.name == "remove_band":
        lo, hi = intervention.low_hz, intervention.high_hz
        if lo is None or hi is None or lo < 0 or hi <= lo:
            raise ValueError("remove_band requires 0 <= low_hz < high_hz")
        nyquist = sample_rate / 2
        # The requested intervention is defined in absolute Hz, while some
        # benchmark recordings have an original Nyquist below 8 kHz.  Apply
        # the intersection with the spectrum that actually exists instead of
        # passing an invalid edge frequency to scipy.signal.butter.
        if lo >= nyquist:
            return x.copy()
        if lo == 0 and hi >= nyquist:
            return np.zeros_like(x)
        if lo == 0:
            return _sos_filter(x, sample_rate, "highpass", hi, order)
        if hi >= nyquist:
            return _sos_filter(x, sample_rate, "lowpass", lo, order)
        return _sos_filter(x, sample_rate, "bandstop", [lo, hi], order)
    raise ValueError(f"unknown intervention: {intervention.name}")


def resample_for_qwen(waveform: np.ndarray, sample_rate: int, target_rate: int = 16000) -> np.ndarray:
    """Polyphase anti-aliased resampling to Qwen's observable 0–8 kHz baseband."""
    x = _mono_float32(waveform)
    if sample_rate == target_rate:
        return x.copy()
    ratio = Fraction(target_rate, sample_rate)
    return resample_poly(x, ratio.numerator, ratio.denominator).astype(np.float32)


def intervention_grid(cutoffs_hz: list[int], bands_hz: list[list[int]]) -> list[Intervention]:
    items = [Intervention("full")]
    items += [Intervention("lowpass", high_hz=f) for f in cutoffs_hz]
    items += [Intervention("remove_band", low_hz=lo, high_hz=hi) for lo, hi in bands_hz]
    return items
