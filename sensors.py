"""
Sensor models for the Sensor Fusion Lab.

Simulates the two sensors every real UAV / missile navigation system has:

* IMU  (100 Hz) -- accelerometer + gyroscope. High rate, but every sample
  carries white noise, and both sensors have a slowly-wandering bias
  ("bias instability"). Integrating them gives smooth motion, but the tiny
  errors accumulate: this is why dead reckoning drifts.
* GPS  (5 Hz)   -- absolute position, noisy (a few meters) and prone to
  dropping out entirely (urban canyons, jamming, "GPS-denied" flight).

All randomness comes from a caller-supplied numpy Generator so every run is
reproducible. No plotting imports -- this module is headless.
"""

import numpy as np


class IMU:
    """MEMS-grade inertial measurement unit.

    Parameters are per-sample standard deviations / bias magnitudes chosen to
    look like a realistic low-cost MEMS IMU (the kind on a small UAV):

    * accel white noise  : 0.02  m/s^2 per sample  (~1200 ug/sqrt(Hz))
    * gyro white noise   : 0.001 rad/s per sample   (~20 deg/hr-ish class)
    * accel bias         : constant ~ N(0, 0.05) m/s^2, drawn once per run
    * gyro bias          : constant ~ N(0, 0.002) rad/s, drawn once per run
    * bias instability   : slow random walk added on top of the constant bias
    """

    def __init__(
        self,
        rng,
        accel_noise=0.02,
        gyro_noise=1e-3,
        accel_bias_std=0.05,
        gyro_bias_std=2e-3,
        bias_rw_step=1e-5,
    ):
        self.rng = rng
        self.accel_noise = accel_noise
        self.gyro_noise = gyro_noise
        # Turn-on biases: unknown constants the navigation filter must learn.
        self.accel_bias = rng.normal(0.0, accel_bias_std, 3)
        self.gyro_bias = rng.normal(0.0, gyro_bias_std, 3)
        # In-run bias instability: a slow random walk around the turn-on bias.
        self.bias_rw_step = bias_rw_step
        self.accel_bias_rw = np.zeros(3)
        self.gyro_bias_rw = np.zeros(3)

    def _step_bias_rw(self):
        self.accel_bias_rw += self.rng.normal(0.0, self.bias_rw_step, 3)
        self.gyro_bias_rw += self.rng.normal(0.0, self.bias_rw_step, 3)

    def measure_accel(self, spec_force_true):
        """Noisy accelerometer reading of the true specific force [m/s^2]."""
        self._step_bias_rw()
        return (
            spec_force_true
            + self.accel_bias
            + self.accel_bias_rw
            + self.rng.normal(0.0, self.accel_noise, 3)
        )

    def measure_gyro(self, gyro_true):
        """Noisy gyroscope reading of the true body rate [rad/s]."""
        # (bias random walk already stepped in measure_accel; IMU sampled together)
        return (
            gyro_true
            + self.gyro_bias
            + self.gyro_bias_rw
            + self.rng.normal(0.0, self.gyro_noise, 3)
        )


class GPS:
    """Noisy absolute position fixes with configurable dropouts.

    std_h / std_v : horizontal / vertical 1-sigma noise [m].
    outages       : iterable of (t_start, t_end) seconds with no fixes at all.
    """

    def __init__(self, rng, std_h=3.0, std_v=5.0, outages=()):
        self.rng = rng
        self.std_h = std_h
        self.std_v = std_v
        self.outages = list(outages)

    def in_outage(self, t):
        return any(t0 <= t < t1 for t0, t1 in self.outages)

    def measure(self, t, pos_true):
        """Return a noisy position fix, or None if GPS is unavailable."""
        if self.in_outage(t):
            return None
        noise = self.rng.normal(
            0.0, [self.std_h, self.std_h, self.std_v]
        )
        return pos_true + noise
