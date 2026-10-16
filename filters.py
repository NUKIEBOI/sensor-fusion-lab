"""
Navigation filters for the Sensor Fusion Lab.

Two filters, two philosophies:

1. ComplementaryFilter -- attitude (roll/pitch/yaw) from gyro + accelerometer.
   A simple, elegant baseline: trust the gyro short-term, trust the gravity
   vector long-term. No magnetometer, so yaw slowly drifts -- a real limitation
   worth seeing.

2. KalmanFilterNav -- the centerpiece. A 9-state linear Kalman filter that
   fuses IMU + GPS for position and velocity:
       x = [px, py, pz, vx, vy, vz, bax, bay, baz]
   The IMU drives the prediction step at 100 Hz; GPS position fixes correct
   the state at 5 Hz. When GPS drops out, the filter coasts on prediction
   alone (this is exactly what real systems do in GPS-denied flight).

Attitude note: the filter needs attitude to rotate body-frame accelerometer
readings into the navigation frame. Here attitude comes from a separate AHRS
(the complementary filter above / ground truth in the demo) -- the same
partition real avionics use (an attitude filter feeding a navigation filter).
"""

import numpy as np

GRAVITY = 9.81


# ---------------------------------------------------------------------------
# Quaternion helpers (q = [w, x, y, z], body -> nav convention)
# ---------------------------------------------------------------------------
def _quat_mult(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def _quat_conj(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def _quat_rotate(q, v):
    """Rotate vector v by quaternion q (body -> nav)."""
    return _quat_mult(_quat_mult(q, np.concatenate([[0.0], v])), _quat_conj(q))[1:]


def _quat_to_rpy(q):
    """Quaternion (body->nav) to roll/pitch/yaw, ZYX convention."""
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


# ---------------------------------------------------------------------------
# 1. Complementary (Mahony-style) attitude filter
# ---------------------------------------------------------------------------
class ComplementaryFilter:
    """Fuses gyro + accelerometer into an attitude estimate.

    How it works: integrate the gyro for a fast, smooth attitude -- but gyro
    bias makes that drift. Separately, the accelerometer measures the specific
    force, which averages out to "up" (minus gravity). Blend them: the gyro
    handles quick motions, the accelerometer slowly pulls the tilt estimate
    back to level. Kp sets how hard the accelerometer corrects; Ki learns the
    gyro bias over time.

    Limitation on purpose: with no magnetometer there is nothing anchoring
    yaw, so yaw drifts. Real AHRS units add a magnetometer for exactly this.
    """

    def __init__(self, dt=0.01, kp=1.0, ki=0.05):
        self.dt = dt
        self.kp = kp
        self.ki = ki
        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # body -> nav
        self.bias = np.zeros(3)

    def update(self, gyro, accel):
        """One filter step. gyro [rad/s], accel = specific force [m/s^2]."""
        # Direction the accelerometer thinks is "up", in body frame.
        f_norm = np.linalg.norm(accel)
        if f_norm < 1e-9:
            return self._quat_to_rpy()
        up_meas = accel / f_norm
        # Direction our current attitude estimate thinks is "up", in body frame.
        up_est = _quat_rotate(_quat_conj(self.q), np.array([0.0, 0.0, 1.0]))
        # Misalignment between the two, as a rotation axis in body frame.
        error = np.cross(up_meas, up_est)
        # PI correction: proportional term fixes tilt now, integral learns bias.
        self.bias += self.ki * error * self.dt
        omega = gyro + self.kp * error + self.bias
        # Integrate attitude: q_dot = 0.5 * q ⊗ [0, omega].
        dq = 0.5 * _quat_mult(self.q, np.concatenate([[0.0], omega * self.dt]))
        self.q = self.q + dq
        self.q /= np.linalg.norm(self.q)
        return self._quat_to_rpy()

    def _quat_to_rpy(self):
        return _quat_to_rpy(self.q)


# ---------------------------------------------------------------------------
# 2. Kalman filter for IMU + GPS navigation
# ---------------------------------------------------------------------------
class KalmanFilterNav:
    """Error-free-formulation linear Kalman filter for position/velocity.

    State:  x = [p(3), v(3), b_a(3)]  (position, velocity, accel bias)

    Predict (100 Hz, IMU-driven):
        f_n = R_nb.T @ (f_b_measured - b_a)   # specific force -> nav frame
        a   = f_n + g                          # acceleration = force + gravity
        p  += v*dt + 0.5*a*dt^2 ;  v += a*dt   # kinematics
        P   = F P F' + Q                        # covariance grows: less sure

    Update (5 Hz, when a GPS fix exists):
        y = z_gps - p ;  S = H P H' + R ;  K = P H' S^-1
        x += K y ;  P = (I - K H) P            # correction: more sure

    During a GPS outage there is simply no update step -- the filter keeps
    predicting. Uncertainty (P) grows while coasting and collapses again
    when fixes return. That is the whole trick of GPS-denied navigation.
    """

    def __init__(self, dt=0.01, sigma_a=0.02, gps_std_h=3.0, gps_std_v=5.0,
                 bias_rw=1e-6):
        self.dt = dt
        self.sigma_a = sigma_a          # accel white noise per-sample std
        self.bias_rw = bias_rw          # accel-bias random-walk PSD
        self.R = np.diag([gps_std_h**2, gps_std_h**2, gps_std_v**2])
        self.g = np.array([0.0, 0.0, -GRAVITY])
        self.x = np.zeros(9)
        self.P = np.eye(9)

    def initialize(self, pos0, vel0):
        self.x[0:3] = pos0
        self.x[3:6] = vel0
        self.x[6:9] = 0.0
        self.P = np.diag([25.0, 25.0, 25.0,     # (5 m)^2 position
                          4.0, 4.0, 4.0,         # (2 m/s)^2 velocity
                          0.0025, 0.0025, 0.0025])  # (0.05 m/s^2)^2 bias

    @property
    def pos(self):
        return self.x[0:3].copy()

    @property
    def vel(self):
        return self.x[3:6].copy()

    def predict(self, f_b_meas, R_nb):
        """IMU-driven prediction. R_nb: nav->body rotation (from the AHRS)."""
        dt = self.dt
        b = self.x[6:9]
        R_bn = R_nb.T  # body->nav: the IMU measures in body axes
        f_n = R_bn @ (f_b_meas - b)
        a = f_n + self.g

        # State prediction (kinematics).
        self.x[0:3] = self.x[0:3] + self.x[3:6] * dt + 0.5 * a * dt * dt
        self.x[3:6] = self.x[3:6] + a * dt
        # (bias states are constant)

        # Covariance prediction.
        I3 = np.eye(3)
        F = np.eye(9)
        F[0:3, 3:6] = dt * I3
        F[0:3, 6:9] = -0.5 * dt * dt * R_bn   # position feels bias via R
        F[3:6, 6:9] = -dt * R_bn              # velocity feels bias via R
        G = np.zeros((9, 3))
        G[0:3, :] = 0.5 * dt * dt * R_bn
        G[3:6, :] = dt * R_bn
        Q = G @ (self.sigma_a**2 * I3) @ G.T
        Q[6:9, 6:9] = self.bias_rw**2 * dt * I3
        self.P = F @ self.P @ F.T + Q

    def update(self, z_gps):
        """GPS position update. Simply not called during outages."""
        H = np.zeros((3, 9))
        H[:, 0:3] = np.eye(3)
        y = z_gps - self.x[0:3]              # innovation: surprise in the fix
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)  # Kalman gain: how much to trust it
        self.x = self.x + K @ y
        # Joseph form: keeps P symmetric positive-definite over thousands of
        # steps (the naive (I-KH)P form slowly loses symmetry and can diverge).
        I_KH = np.eye(9) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T


def dead_reckoning(f_b_meas_seq, R_nb_seq, dt, pos0, vel0):
    """Pure inertial navigation: integrate the IMU, correct for nothing.

    This is the baseline that shows why you can't navigate on an IMU alone:
    the unknown accelerometer bias integrates twice into position, so the
    error grows ~ quadratically with time.
    """
    g = np.array([0.0, 0.0, -GRAVITY])
    p = pos0.copy().astype(float)
    v = vel0.copy().astype(float)
    hist = np.empty_like(f_b_meas_seq)
    for k, (f_b, R_nb) in enumerate(zip(f_b_meas_seq, R_nb_seq)):
        a = R_nb.T @ f_b + g  # body->nav rotation
        v = v + a * dt
        p = p + v * dt
        hist[k] = p
    return hist
