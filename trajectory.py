"""
True trajectory generation for the Sensor Fusion Lab.

Generates a smooth, UAV-style flight path -- a surveillance circuit with
straight legs, coordinated turns, and altitude changes -- and produces the
ground-truth quantities a navigation system would try to estimate:

    position, velocity, acceleration      (navigation frame)
    attitude (roll / pitch / yaw)         (navigation -> body)
    true gyroscope readings               (body angular rate, rad/s)
    true accelerometer readings           (specific force in body frame, m/s^2)

Frames
------
* Navigation frame: ENU (x = East, y = North, z = Up), z-up.
* Body frame: x = forward, y = left, z = up  (right-handed: fwd x left = up).

The attitude is derived from the velocity vector the way a real fixed-wing
UAV flies it: yaw follows the ground track, pitch follows the flight-path
angle, and roll banks into turns (coordinated turn, ball centered).

The aircraft flies the circuit at a constant cruise speed, so the dynamics
stay realistic no matter how long the scenario runs.
"""

import numpy as np

GRAVITY = 9.81  # m/s^2 magnitude; nav-frame gravity vector is [0, 0, -g]
CRUISE_SPEED = 18.0  # m/s, typical small-UAV cruise


# ---------------------------------------------------------------------------
# Path smoothing helpers
# ---------------------------------------------------------------------------
def _catmull_rom_closed(points, samples_per_seg=400):
    """Uniform Catmull-Rom spline through `points`, treated as a closed loop."""
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    out = []
    for i in range(n):
        p0 = pts[(i - 1) % n]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n]
        for j in range(samples_per_seg):
            t = j / samples_per_seg
            t2, t3 = t * t, t * t * t
            pos = 0.5 * (
                2 * p1
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t3
            )
            out.append(pos)
    return np.array(out)


def _arclength_grid(pts):
    """Arclength coordinate s[i] for each point of an open polyline."""
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def rotation_nav_to_body(roll, pitch, yaw):
    """Rotation matrix taking nav-frame (ENU) vectors into the body frame.

    ZYX Euler convention with yaw measured counter-clockwise from East
    (standard math angle in the ENU plane): R_nb = Rx(roll) @ Ry(pitch) @ Rz(-yaw).
    The minus sign is needed because a CCW-positive yaw is the *inverse* of the
    usual clockwise-positive aerospace yaw.
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(-yaw), np.sin(-yaw)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rx @ Ry @ Rz


def _moving_average(x, window=11):
    """Light centered smoothing (for numerical derivatives)."""
    w = np.ones(window) / window
    return np.apply_along_axis(lambda m: np.convolve(m, w, mode="same"), 0, x)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------
def generate_trajectory(duration=120.0, dt=0.01, cruise_speed=CRUISE_SPEED):
    """Build the true trajectory.

    The UAV flies a closed waypoint circuit at constant cruise speed,
    looping as many times as needed to fill `duration` seconds.

    Returns a dict with keys:
        t          (N,)      time stamps [s]
        pos        (N, 3)    true position, ENU [m]
        vel        (N, 3)    true velocity [m/s]
        acc        (N, 3)    true acceleration [m/s^2]
        rpy        (N, 3)    true roll, pitch, yaw [rad]
        R_nb       (N, 3, 3) true nav->body rotation matrices
        gyro       (N, 3)    true body angular rate [rad/s]
        spec_force (N, 3)    true specific force in body frame [m/s^2]
        gps_idx    (M,)      sample indices of GPS epochs (5 Hz)
    """
    n = int(duration / dt)
    t = np.arange(n) * dt

    # Waypoints: closed surveillance circuit, ~700 x 600 m, altitude 90-115 m.
    # Listed counter-clockwise seen from above (all left turns).
    waypoints = [
        (0.0, 0.0, 100.0),
        (350.0, 0.0, 100.0),
        (550.0, 150.0, 115.0),
        (550.0, 450.0, 115.0),
        (350.0, 600.0, 100.0),
        (0.0, 600.0, 100.0),
        (-150.0, 450.0, 90.0),
        (-150.0, 150.0, 90.0),
    ]

    loop = _catmull_rom_closed(waypoints, samples_per_seg=400)
    lap_length = _arclength_grid(np.vstack([loop, loop[0]]))[-1]

    # Tile the loop so we can fly partial or multiple laps at cruise speed.
    n_laps = int(np.ceil(cruise_speed * duration / lap_length)) + 1
    tiled = np.tile(loop, (n_laps, 1))
    s_grid = _arclength_grid(tiled)
    s_target = np.linspace(0.0, cruise_speed * duration, n)
    pos = np.column_stack(
        [np.interp(s_target, s_grid, tiled[:, k]) for k in range(3)]
    )

    # Velocity / acceleration by central differences, lightly smoothed.
    vel = np.gradient(pos, dt, axis=0)
    acc = _moving_average(np.gradient(vel, dt, axis=0), window=11)

    speed = np.linalg.norm(vel, axis=1)
    speed = np.maximum(speed, 1e-6)

    # Attitude from the velocity vector (how a fixed-wing UAV actually flies).
    yaw = np.unwrap(np.arctan2(vel[:, 1], vel[:, 0]))
    yaw_rate = np.gradient(yaw, dt)
    v_horiz = np.hypot(vel[:, 0], vel[:, 1])
    pitch = np.arctan2(vel[:, 2], np.maximum(v_horiz, 1e-6))
    pitch = _moving_average(pitch[:, None], window=21)[:, 0]
    # Coordinated turn: bank into the turn so lateral specific force ~ 0.
    # (Sign verified on a constant-radius circle: this choice zeroes the
    #  lateral accelerometer reading exactly in steady turns.)
    roll = np.arctan2(speed * yaw_rate, GRAVITY)
    roll = _moving_average(roll[:, None], window=21)[:, 0]

    rpy = np.column_stack([roll, pitch, yaw])
    R_nb = np.array([rotation_nav_to_body(r, p, y) for r, p, y in rpy])

    # True body angular rate from R_nb:  skew(w) = dR/dt @ R^T.
    dR = np.gradient(R_nb, dt, axis=0)
    gyro = np.empty((n, 3))
    for k in range(n):
        W = dR[k] @ R_nb[k].T
        gyro[k] = [W[2, 1], W[0, 2], W[1, 0]]  # vee operator of skew matrix

    # True specific force (what an ideal accelerometer feels):
    #   f = a - g,  expressed in the body frame.
    g_nav = np.array([0.0, 0.0, -GRAVITY])
    spec_force = np.einsum("nij,nj->ni", R_nb, acc - g_nav)

    # Sanity: coordinated flight => lateral (y) specific force near zero.
    if np.median(np.abs(spec_force[:, 1])) > 0.5:
        raise RuntimeError("Turn coordination looks wrong (check roll sign).")

    gps_idx = np.arange(0, n, int(round(0.2 / dt)))  # 5 Hz GPS epochs

    return {
        "t": t,
        "pos": pos,
        "vel": vel,
        "acc": acc,
        "rpy": rpy,
        "R_nb": R_nb,
        "gyro": gyro,
        "spec_force": spec_force,
        "gps_idx": gps_idx,
        "dt": dt,
    }
