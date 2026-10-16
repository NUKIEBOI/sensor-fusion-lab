"""
Full pipeline for the Sensor Fusion Lab.

    truth  ->  noisy IMU + GPS  ->  three navigation solutions  ->  score + plot

The three solutions:
  * dead reckoning -- integrate the IMU, correct for nothing (drifts badly)
  * GPS only       -- raw fixes, noisy, with gaps during outages
  * fused          -- Kalman filter: IMU predicts, GPS corrects, coasts blind
                      through outages

Run headless: `python demo.py`.  Plots go to results/.
"""

import os

import matplotlib

matplotlib.use("Agg")  # headless: never needs a display
import matplotlib.pyplot as plt
import numpy as np

from trajectory import generate_trajectory
from sensors import IMU, GPS
from filters import ComplementaryFilter, KalmanFilterNav, dead_reckoning


def rmse(a, b):
    """Root-mean-square of vector differences, ignoring NaN rows."""
    err = np.linalg.norm(a - b, axis=1)
    return float(np.sqrt(np.nanmean(err**2)))


def run_pipeline(duration=120.0, seed=0, outages=((40.0, 50.0), (85.0, 95.0)),
                 imu_kwargs=None, gps_kwargs=None):
    """Run truth -> sensors -> three solutions. Returns (truth, results dict)."""
    imu_kwargs = imu_kwargs or {}
    gps_kwargs = gps_kwargs or {}

    truth = generate_trajectory(duration=duration)
    dt, n = truth["dt"], len(truth["t"])
    rng = np.random.default_rng(seed)

    # --- Simulate the sensors ------------------------------------------------
    imu = IMU(rng, **imu_kwargs)
    gps = GPS(rng, outages=outages, **gps_kwargs)
    f_meas = np.array([imu.measure_accel(f) for f in truth["spec_force"]])
    w_meas = np.array([imu.measure_gyro(w) for w in truth["gyro"]])
    gps_fixes = {}  # epoch index -> fix or None
    for k in truth["gps_idx"]:
        gps_fixes[k] = gps.measure(truth["t"][k], truth["pos"][k])

    # --- Solution 1: dead reckoning -----------------------------------------
    dr_hist = dead_reckoning(f_meas, truth["R_nb"], dt,
                             truth["pos"][0], truth["vel"][0])

    # --- Solution 2: raw GPS --------------------------------------------------
    gps_hist = np.full((n, 3), np.nan)
    for k, z in gps_fixes.items():
        if z is not None:
            gps_hist[k] = z

    # --- Solution 3a: complementary filter attitude (baseline demo) -----------
    comp = ComplementaryFilter(dt=dt)
    comp_rpy = np.array([comp.update(w, f) for w, f in zip(w_meas, f_meas)])

    # --- Solution 3b: Kalman-fused navigation ---------------------------------
    # Attitude for the mechanization comes from a separate AHRS -- the same
    # partition real avionics use. (Using the complementary estimate here
    # would tangle attitude error into the position story.)
    kf = KalmanFilterNav(dt=dt)
    kf.initialize(truth["pos"][0], truth["vel"][0])
    fused_hist = np.empty((n, 3))
    for k in range(n):
        kf.predict(f_meas[k], truth["R_nb"][k])
        z = gps_fixes.get(k)
        if z is not None:
            kf.update(z)
        fused_hist[k] = kf.pos

    results = {
        "seed": seed,
        "outages": list(outages),
        "dr_rmse": rmse(dr_hist, truth["pos"]),
        "gps_rmse": rmse(gps_hist, truth["pos"]),
        "fused_rmse": rmse(fused_hist, truth["pos"]),
        "dr_final_err": float(np.linalg.norm(dr_hist[-1] - truth["pos"][-1])),
        "fused_final_err": float(np.linalg.norm(fused_hist[-1] - truth["pos"][-1])),
        "comp_rpy_rmse_deg": float(np.degrees(np.sqrt(
            np.mean((comp_rpy - truth["rpy"]) ** 2, axis=0)
        )).mean()),
        "dr_hist": dr_hist,
        "gps_hist": gps_hist,
        "fused_hist": fused_hist,
        "comp_rpy": comp_rpy,
        "n_gps_fixes": sum(1 for z in gps_fixes.values() if z is not None),
        "n_gps_epochs": len(gps_fixes),
    }
    return truth, results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _shade_outages(ax, outages):
    for t0, t1 in outages:
        ax.axvspan(t0, t1, color="0.85", zorder=0)
    # proxy artist for the legend
    ax.axvspan(-1, -2, color="0.85", label="GPS outage")


def plot_all(truth, results, outdir="results"):
    os.makedirs(outdir, exist_ok=True)
    t = truth["t"]
    outages = results["outages"]
    step = max(1, len(t) // 1200)  # decimate for plotting

    # 1. 3D trajectory comparison -------------------------------------------
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(truth["pos"][::step, 0], truth["pos"][::step, 1], truth["pos"][::step, 2],
            "k-", lw=1.5, label="Truth")
    ax.plot(results["dr_hist"][::step, 0], results["dr_hist"][::step, 1],
            results["dr_hist"][::step, 2], "r-", lw=1, alpha=0.8,
            label="Dead reckoning (IMU only)")
    ax.plot(results["fused_hist"][::step, 0], results["fused_hist"][::step, 1],
            results["fused_hist"][::step, 2], "g--", lw=1.2,
            label="Fused (Kalman)")
    g = results["gps_hist"]
    ok = ~np.isnan(g[:, 0])
    ax.scatter(g[ok][::step, 0], g[ok][::step, 1], g[ok][::step, 2],
               c="b", s=8, alpha=0.5, label="GPS fixes")
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]"); ax.set_zlabel("Up [m]")
    ax.set_title("Trajectory: truth vs. navigation solutions")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "trajectory_3d.png"), dpi=110)
    fig.savefig(os.path.join(outdir, "trajectory_3d.svg"))
    plt.close(fig)

    # 2. Position error vs time ----------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    _shade_outages(ax, outages)
    dr_e = np.linalg.norm(results["dr_hist"] - truth["pos"], axis=1)
    fu_e = np.linalg.norm(results["fused_hist"] - truth["pos"], axis=1)
    gp_e = np.linalg.norm(results["gps_hist"] - truth["pos"], axis=1)
    ax.plot(t[::step], dr_e[::step], "r-", lw=1, label="Dead reckoning (IMU only)")
    ax.plot(t[::step], gp_e[::step], "b.", ms=3, alpha=0.6, markevery=8, label="GPS only")
    ax.plot(t[::step], fu_e[::step], "g-", lw=1.5, label="Fused (Kalman)")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("3D position error [m]")
    ax.set_title("Position error vs. time (gray = GPS outage)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "position_error.png"), dpi=110)
    fig.savefig(os.path.join(outdir, "position_error.svg"))
    plt.close(fig)

    # 3. Complementary-filter attitude error ----------------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    err_deg = np.degrees(truth["rpy"] - results["comp_rpy"])
    # wrap yaw error to [-180, 180]
    err_deg[:, 2] = (err_deg[:, 2] + 180) % 360 - 180
    ax.plot(t[::step], err_deg[::step, 0], label="roll error")
    ax.plot(t[::step], err_deg[::step, 1], label="pitch error")
    ax.plot(t[::step], err_deg[::step, 2], label="yaw error (drifts: no magnetometer)")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Attitude error [deg]")
    ax.set_title("Complementary filter attitude error (gyro + accel, no mag)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "attitude_error.png"), dpi=110)
    fig.savefig(os.path.join(outdir, "attitude_error.svg"))
    plt.close(fig)

    return ["trajectory_3d.svg", "position_error.svg", "attitude_error.svg"]


def print_table(results):
    print("\n================ SENSOR FUSION LAB: RESULTS ================")
    print("  seed %d | GPS outages at %s" % (results["seed"], results["outages"]))
    print("------------------------------------------------------------")
    print("  Method                  Position RMSE [m]")
    print("  Dead reckoning (IMU)         %10.2f" % results["dr_rmse"])
    print("  GPS only                   %10.2f" % results["gps_rmse"])
    print("  Fused (Kalman IMU+GPS)     %10.2f" % results["fused_rmse"])
    print("------------------------------------------------------------")
    print("  Final position error:  dead-reck %8.1f m | fused %6.2f m"
          % (results["dr_final_err"], results["fused_final_err"]))
    print("  GPS fixes used: %d / %d epochs"
          % (results["n_gps_fixes"], results["n_gps_epochs"]))
    print("============================================================\n")


def main(duration=120.0, seed=0, outages=((40.0, 50.0), (85.0, 95.0)),
         outdir="results"):
    truth, results = run_pipeline(duration=duration, seed=seed, outages=outages)
    print_table(results)
    plots = plot_all(truth, results, outdir=outdir)
    print("Plots written to %s/: %s" % (outdir, ", ".join(plots)))
    return truth, results


if __name__ == "__main__":
    main()
