import numpy as np

from animal_omni.audio import Intervention, apply_intervention, resample_for_qwen


def _power_at(x, sr, hz):
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(len(x), 1 / sr)
    return spectrum[np.argmin(abs(freqs - hz))]


def test_lowpass_suppresses_high_frequency_before_resampling():
    sr = 96000
    t = np.arange(sr) / sr
    x = np.sin(2*np.pi*2000*t) + np.sin(2*np.pi*12000*t)
    y = apply_intervention(x, sr, Intervention("lowpass", high_hz=4000))
    assert _power_at(y, sr, 2000) > 1e5 * _power_at(y, sr, 12000)
    assert len(resample_for_qwen(y, sr)) == 16000


def test_band_removal_is_selective():
    sr = 16000
    t = np.arange(sr) / sr
    x = sum(np.sin(2*np.pi*f*t) for f in (500, 1500, 3000))
    y = apply_intervention(x, sr, Intervention("remove_band", 1000, 2000))
    assert _power_at(y, sr, 500) > 1e4 * _power_at(y, sr, 1500)
    assert _power_at(y, sr, 3000) > 1e4 * _power_at(y, sr, 1500)


def test_lowpass_above_original_nyquist_is_noop():
    x = np.linspace(-1, 1, 128, dtype=np.float32)
    y = apply_intervention(x, 1280, Intervention("lowpass", high_hz=1000), order=4)
    assert np.array_equal(x, y)


def test_band_removal_clips_to_original_nyquist():
    sr = 16000
    t = np.arange(sr) / sr
    x = np.sin(2*np.pi*3000*t) + np.sin(2*np.pi*7000*t)
    y = apply_intervention(x, sr, Intervention("remove_band", 6000, 8000))
    assert _power_at(y, sr, 3000) > 1e4 * _power_at(y, sr, 7000)


def test_band_removal_above_nyquist_is_noop_and_full_cover_is_zero():
    x = np.linspace(-1, 1, 128, dtype=np.float32)
    above = apply_intervention(x, 8000, Intervention("remove_band", 6000, 8000), order=4)
    covered = apply_intervention(x, 8000, Intervention("remove_band", 0, 4000), order=4)
    assert np.array_equal(x, above)
    assert np.array_equal(np.zeros_like(x), covered)
