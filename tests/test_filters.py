"""
Tests for the Sensor Fusion Lab.

Run headless:  python tests/test_filters.py

The four claims the project makes, checked with plain asserts:
  1. The fused solution beats raw GPS on position RMSE.
  2. The fused solution beats dead reckoning by a wide margin.
  3. The filter survives a long GPS outage: no NaN, error stays bounded.
  4. With (near-)zero sensor noise the filter tracks truth almost exactly.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from demo import run_pipeline


def _check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print("  [%s] %s %s" % (status, name, detail))
    if not cond:
        raise AssertionError("FAILED: " + name)


def test_fused_beats_gps():
    _, r = run_pipeline(duration=60.0, seed=1, outages=((25.0, 35.0),))
    _check("fused RMSE < GPS-only RMSE",
           r["fused_rmse"] < r["gps_rmse"],
           "(fused %.2f m < gps %.2f m)" % (r["fused_rmse"], r["gps_rmse"]))


def test_fused_beats_dead_reckoning():
    _, r = run_pipeline(duration=60.0, seed=2, outages=((25.0, 35.0),))
    _check("fused RMSE < 20% of dead-reckoning RMSE",
           r["fused_rmse"] < 0.2 * r["dr_rmse"],
           "(fused %.2f m << dead-reck %.1f m)" % (r["fused_rmse"], r["dr_rmse"]))


def test_survives_long_outage():
    truth, r = run_pipeline(duration=60.0, seed=3, outages=((20.0, 35.0),))
    fused = r["fused_hist"]
    _check("no NaN/inf through 15 s outage",
           np.all(np.isfinite(fused)), "")
    # Worst error *during* the outage window should stay bounded.
    in_out = (truth["t"] >= 20.0) & (truth["t"] < 35.0)
    worst = float(np.linalg.norm(fused[in_out] - truth["pos"][in_out], axis=1).max())
    _check("error stays bounded through 15 s outage",
           worst < 60.0,
           "(worst in-outage error %.1f m)" % worst)


def test_zero_noise_tracks_truth():
    _, r = run_pipeline(
        duration=30.0, seed=4, outages=(),
        imu_kwargs=dict(accel_noise=0.0, gyro_noise=0.0,
                        accel_bias_std=0.0, gyro_bias_std=0.0,
                        bias_rw_step=0.0),
        gps_kwargs=dict(std_h=1e-3, std_v=1e-3),
    )
    _check("zero-noise fused RMSE < 0.05 m",
           r["fused_rmse"] < 0.05,
           "(fused RMSE %.4f m)" % r["fused_rmse"])


if __name__ == "__main__":
    print("Running sensor-fusion-lab tests...")
    test_fused_beats_gps()
    test_fused_beats_dead_reckoning()
    test_survives_long_outage()
    test_zero_noise_tracks_truth()
    print("All tests passed.")
