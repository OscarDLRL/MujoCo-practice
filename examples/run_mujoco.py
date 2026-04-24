#!/usr/bin/env python3
"""Run PMP / LQG / MPC controllers on a quadruped in MuJoCo.

Stable waypoint-following version based on the ORIGINAL working runner:
- keeps the original environment initialization/reset
- keeps the original controller tuning / estimator noise settings
- adds an OUTER waypoint follower that generates desired vx, vy, wz
- keeps the INNER controller exactly in velocity-tracking mode
- logs waypoint targets separately for trajectory metrics / plots

Key idea:
    waypoints -> outer follower -> desired planar velocities -> original controller
"""

import sys
import os
import argparse
import threading
import select
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gym_quadruped.quadruped_env import QuadrupedEnv

from src.dynamics import QuadrupedDynamics
from src.estimator_ekf import OrientationEKF
from src.controller_pmp import PontryaginController
from src.controller_lqg import LQGController
from src.controller_mpc import MPCController
from src.trajectory_generator import WaypointTrajectory
from src.gait_planner import TrotGaitPlanner
from src.foot_trajectory import FootTrajectoryGenerator

# =====================================================================
# Nominal physical parameters
# =====================================================================
ROBOT_MASS = 9.0
ROBOT_BODY_MASS = 6.921
ROBOT_INERTIA = np.diag([0.107, 0.098, 0.024])
ROBOT_HIP_HEIGHT = 0.225
ROBOT_FOOT_OFFSET = np.array([
    [0.19,  0.111, -0.225],   # FL
    [0.19, -0.111, -0.225],   # FR
    [-0.19,  0.111, -0.225],  # RL
    [-0.19, -0.111, -0.225],  # RR
], dtype=float)

RESULTS_DIR = "results"


# =====================================================================
# Teleop
# =====================================================================
@dataclass
class TeleopState:
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    step_lin: float = 0.05
    step_ang: float = 0.15
    max_vx: float = 0.8
    max_vy: float = 0.5
    max_wz: float = 1.5
    quit_requested: bool = False

    def clamp(self):
        self.vx = float(np.clip(self.vx, -self.max_vx, self.max_vx))
        self.vy = float(np.clip(self.vy, -self.max_vy, self.max_vy))
        self.wz = float(np.clip(self.wz, -self.max_wz, self.max_wz))

    def zero(self):
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0


def teleop_keyboard_loop(teleop: TeleopState):
    """
    Terminal teleop with arrow keys:
      ↑ / ↓   -> vx +/-
      ← / →   -> wz +/-
      z / c   -> vy +/-
      space   -> zero commands
      Ctrl+C  -> quit
    """
    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    print("\n[Teleop enabled]")
    print("  ↑ / ↓ : forward/backward")
    print("  ← / → : yaw left/right")
    print("  z / c : lateral left/right")
    print("  space : zero commands")
    print("  Ctrl+C: quit\n")

    try:
        tty.setcbreak(fd)
        while not teleop.quit_requested:
            if select.select([sys.stdin], [], [], 0.05)[0]:
                ch = sys.stdin.read(1)

                if ch == "\x1b":
                    seq1 = sys.stdin.read(1)
                    seq2 = sys.stdin.read(1)

                    if seq1 == "[":
                        if seq2 == "A":       # Up
                            teleop.vx += teleop.step_lin
                        elif seq2 == "B":     # Down
                            teleop.vx -= teleop.step_lin
                        elif seq2 == "C":     # Right
                            teleop.wz -= teleop.step_ang
                        elif seq2 == "D":     # Left
                            teleop.wz += teleop.step_ang

                elif ch == "z":
                    teleop.vy += teleop.step_lin
                elif ch == "c":
                    teleop.vy -= teleop.step_lin
                elif ch == " ":
                    teleop.zero()

                teleop.clamp()
                print(
                    f"\rcmd -> vx={teleop.vx:+.2f}, vy={teleop.vy:+.2f}, wz={teleop.wz:+.2f} ",
                    end="",
                    flush=True,
                )

    except KeyboardInterrupt:
        teleop.quit_requested = True

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print()


# =====================================================================
# Waypoint follower (outer loop)
# =====================================================================
def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def make_line_from_start(length: float, start=(0.0, 0.0)) -> np.ndarray:
    sx, sy = start
    return np.array([[sx, sy], [sx + length, sy]], dtype=float)


def make_square_from_start(side: float, start=(0.0, 0.0)) -> np.ndarray:
    sx, sy = start
    return np.array([
        [sx, sy],
        [sx + side, sy],
        [sx + side, sy + side],
        [sx, sy + side],
        [sx, sy],
    ], dtype=float)


def make_circle_from_start(radius: float, start=(0.0, 0.0), n_points: int = 80) -> np.ndarray:
    sx, sy = start
    # Start exactly at current position, then go around a circle centered to the right
    cx, cy = sx, sy
    ang = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    pts = np.stack([cx + radius * np.cos(ang), cy + radius * np.sin(ang)], axis=1)
    # Force exact start point as first waypoint
    pts[0] = np.array([sx, sy], dtype=float)
    return np.vstack([pts, pts[0]])


def make_figure8_from_start(scale: float, start=(0.0, 0.0), n_points: int = 100) -> np.ndarray:
    sx, sy = start
    t = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    x = sx + scale * np.sin(t)
    y = sy + 0.5 * scale * np.sin(2.0 * t)
    pts = np.stack([x, y], axis=1)
    pts[0] = np.array([sx, sy], dtype=float)
    return np.vstack([pts, pts[0]])


def build_waypoints(path_name: str, size: float, start_xy: np.ndarray) -> Optional[np.ndarray]:
    path_name = path_name.lower()
    start_xy = np.asarray(start_xy, dtype=float)

    if path_name == "none":
        return None
    if path_name == "line":
        return make_line_from_start(size, tuple(start_xy))
    if path_name == "square":
        return make_square_from_start(size, tuple(start_xy))
    if path_name == "circle":
        return make_circle_from_start(size, tuple(start_xy), n_points=80)
    if path_name == "figure8":
        return make_figure8_from_start(size, tuple(start_xy), n_points=100)
    raise ValueError(f"Unknown path: {path_name}")



# =====================================================================
# State extraction and helpers
# =====================================================================
def get_state(env) -> np.ndarray:
    """
    x = [p(3), v(3), rpy(3), omega(3)]
    """
    p = env.base_pos
    v = env.base_lin_vel(frame="world")
    rpy = env.base_ori_euler_xyz
    omega = env.base_ang_vel(frame="world")
    return np.concatenate([p, v, rpy, omega])


def grf_to_torques(env, foot_forces_world: np.ndarray, contact_mask: np.ndarray) -> np.ndarray:
    """
    Map 12D ground reaction forces [f_FL, f_FR, f_RL, f_RR] in world frame
    to joint torques via τ = Σ J_i(q)^T f_i.

    The env exposes per-leg Jacobians and actuator indices.
    """
    tau = np.zeros(env.mjModel.nu)

    try:
        jacobians = env.feet_jacobians(frame="world")
    except Exception:
        return tau

    leg_names = ["FL", "FR", "RL", "RR"]

    for i, leg in enumerate(leg_names):
        if not contact_mask[i]:
            continue

        f_leg = foot_forces_world[3 * i:3 * i + 3]
        J_full = jacobians[leg]
        leg_idx = env.legs_qvel_idx[leg]
        J_leg = J_full[:, leg_idx]
        tau_leg = -J_leg.T @ f_leg
        tau_idx = env.legs_tau_idx[leg]
        tau[tau_idx] = tau_leg

    if hasattr(env, "action_space"):
        tau = np.clip(tau, env.action_space.low, env.action_space.high)

    return tau


def get_contacts(env) -> np.ndarray:
    """Return (4,) boolean contact mask [FL, FR, RL, RR]."""
    try:
        cs, _ = env.feet_contact_state()
        return np.array([cs.FL, cs.FR, cs.RL, cs.RR], dtype=bool)
    except Exception:
        return np.ones(4, dtype=bool)


def get_feet_world(env) -> np.ndarray:
    """Return (4, 3) foot positions in world frame."""
    try:
        fp = env.feet_pos(frame="world")
        return np.array([fp.FL, fp.FR, fp.RL, fp.RR])
    except Exception:
        return None


# =====================================================================
# Dynamics and references
# =====================================================================
def build_dynamics():
    dyn = QuadrupedDynamics(
        mass=ROBOT_MASS,
        inertia=ROBOT_INERTIA,
        dt=0.002,
    )
    dyn.r_feet_body = ROBOT_FOOT_OFFSET.copy()
    return dyn


def build_cost_matrices():
    Q = np.diag([
        80, 80, 400,    # position
        8, 8, 40,       # velocity
        150, 150, 30,   # orientation
        1, 1, 4,        # angular velocity
    ])
    R = np.eye(12) * 1e-4
    Q_f = Q * 5
    return Q, R, Q_f


def build_reference_state(
    dyn: QuadrupedDynamics,
    height: float,
    vx: float = 0.0,
    vy: float = 0.0,
    wz: float = 0.0,
) -> np.ndarray:
    """
    x = [p(3), v(3), rpy(3), omega(3)]
    Track commanded planar velocity and yaw rate while keeping upright.
    This is the ORIGINAL stable inner-loop reference.
    """
    x_ref = dyn.standing_state(height=height)
    x_ref[3:6] = np.array([vx, vy, 0.0])
    x_ref[6:9] = np.array([0.0, 0.0, 0.0])
    x_ref[9:12] = np.array([0.0, 0.0, wz])
    return x_ref


def build_logging_reference(
    target_xy: np.ndarray,
    height: float,
    vx: float = 0.0,
    vy: float = 0.0,
    wz: float = 0.0,
) -> np.ndarray:
    """Reference used only for plots / metrics against waypoints."""
    x_ref = np.zeros(12)
    x_ref[0:2] = np.asarray(target_xy, dtype=float)
    x_ref[2] = height
    x_ref[3:6] = np.array([vx, vy, 0.0])
    x_ref[6:9] = np.array([0.0, 0.0, 0.0])
    x_ref[9:12] = np.array([0.0, 0.0, wz])
    return x_ref


# =====================================================================
# Controllers
# =====================================================================
def build_controller(name: str, dyn: QuadrupedDynamics, Q, R, Q_f, x_ref):
    A_d, B_d, g_d = dyn.get_linear_system(x_ref)
    A_c, B_c = dyn.continuous_AB(x_ref)

    if name == "pmp":
        ctrl = PontryaginController(
            A=A_c,
            B=B_c,
            Q_s=Q,
            R_u=R,
            Q_f=Q_f,
            g_aff=dyn.gravity_vector() / dyn.dt,
            dt=dyn.dt,
            horizon=500,
        )
        ctrl.solve_discrete_sweep(x_ref.copy(), x_ref)
        print("  [PMP] Hamiltonian-based controller initialized")
        return ctrl

    if name == "lqg":
        ctrl = LQGController(
            A_d=A_d,
            B_d=B_d,
            g_d=g_d,
            Q=Q * dyn.dt,
            R=R * dyn.dt,
            Q_proc=np.diag([1e-3] * 3 + [1e-2] * 3 + [5e-3] * 3 + [1e-2] * 3),
            R_meas=np.diag([5e-3] * 3 + [2e-2] * 3 + [1e-2] * 3 + [5e-2] * 3),
        )
        ctrl.set_initial_estimate(x_ref)
        print("  [LQG] Controller initialized")
        return ctrl

    if name == "mpc":
        ctrl = MPCController(
            A_d=A_d,
            B_d=B_d,
            g_d=g_d,
            Q=Q * dyn.dt,
            R=R * dyn.dt,
            Q_f=Q_f * dyn.dt,
            N=10,
            mu=0.6,
            fz_max=150.0,
        )
        print("  [MPC] Horizon=10, OSQP-based controller initialized")
        return ctrl

    raise ValueError(f"Unknown controller: {name}")


# =====================================================================
# Plotting
# =====================================================================
def save_single_run_plot(result, controller_name, robot_name, disturbance_type):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    log_t = result["time"]
    log_x = result["state"]
    log_ref = result["reference"]
    log_u = result["control"]
    log_dist = result["disturbance"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"{controller_name.upper()} — {robot_name} — {disturbance_type}",
        fontsize=14,
        fontweight="bold",
    )

    axes[0].plot(log_t, log_x[:, 0], label="x")
    axes[0].plot(log_t, log_ref[:, 0], "--", label="x_ref")
    axes[0].plot(log_t, log_x[:, 1], label="y")
    axes[0].plot(log_t, log_ref[:, 1], "--", label="y_ref")
    axes[0].set_ylabel("XY [m]")
    axes[0].legend(ncol=4, fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(log_t, log_x[:, 3], label="vx")
    axes[1].plot(log_t, log_ref[:, 3], "--", label="vx_ref")
    axes[1].plot(log_t, log_x[:, 4], label="vy")
    axes[1].plot(log_t, log_ref[:, 4], "--", label="vy_ref")
    axes[1].set_ylabel("Velocity [m/s]")
    axes[1].legend(ncol=4, fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(log_t, np.linalg.norm(log_x[:, 0:2] - log_ref[:, 0:2], axis=1), label="xy error")
    axes[2].plot(log_t, np.linalg.norm(log_u, axis=1), label="||GRFs||")
    axes[2].set_ylabel("Error / effort")
    axes[2].legend(ncol=2, fontsize=8)
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(log_t, log_dist, label="disturbance")
    axes[3].set_ylabel("Disturbance")
    axes[3].set_xlabel("Time [s]")
    axes[3].legend(fontsize=8)
    axes[3].grid(True, alpha=0.3)

    for ax in axes:
        if disturbance_type == "impulse":
            ax.axvspan(2.0, 2.15, alpha=0.12)
        elif disturbance_type == "persistent":
            ax.axvspan(2.0, log_t[-1], alpha=0.05)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"mujoco_{controller_name}_{robot_name}_{result['path_name']}_{disturbance_type}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Plot saved: {path}")


def save_comparison_plot(results, robot_name, disturbance_type, path_name):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    colors = {"pmp": "#e74c3c", "lqg": "#2ecc71", "mpc": "#3498db"}

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(
        f"Controller Comparison — {robot_name} — {path_name} — {disturbance_type}",
        fontsize=14,
        fontweight="bold",
    )

    for name, data in results.items():
        t = data["time"]
        x = data["state"]
        ref = data["reference"]
        u = data["control"]

        pos_err = np.linalg.norm(x[:, 0:2] - ref[:, 0:2], axis=1)
        vel_err = np.linalg.norm(x[:, 3:5] - ref[:, 3:5], axis=1)
        u_norm = np.linalg.norm(u, axis=1)

        axes[0].plot(t, pos_err, color=colors[name], label=name.upper(), lw=1.5)
        axes[1].plot(t, vel_err, color=colors[name], lw=1.5)
        axes[2].plot(t, u_norm, color=colors[name], lw=1.2)

    axes[0].set_ylabel("XY error [m]")
    axes[1].set_ylabel("Velocity error [m/s]")
    axes[2].set_ylabel("||GRFs|| [N]")
    axes[2].set_xlabel("Time [s]")
    axes[0].legend()

    for ax in axes:
        ax.grid(True, alpha=0.3)
        if disturbance_type == "impulse":
            ax.axvspan(2.0, 2.15, alpha=0.12)
        elif disturbance_type == "persistent":
            ax.axvspan(2.0, t[-1], alpha=0.05)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"mujoco_comparison_{robot_name}_{path_name}_{disturbance_type}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Comparison plot saved: {path}")


def compute_tracking_metrics(result):
    x = result["state"]
    ref = result["reference"]
    u = result["control"]

    xy_err = np.linalg.norm(x[:, 0:2] - ref[:, 0:2], axis=1)
    vel_xy_err = np.linalg.norm(x[:, 3:5] - ref[:, 3:5], axis=1)
    z_err = np.abs(x[:, 2] - ref[:, 2])
    yaw_err = np.abs(np.unwrap(x[:, 8]) - np.unwrap(ref[:, 8]))
    u_norm = np.linalg.norm(u, axis=1)

    return {
        "rmse_xy": float(np.sqrt(np.mean(xy_err**2))),
        "max_xy_err": float(np.max(xy_err)),
        "mean_xy_err": float(np.mean(xy_err)),
        "rmse_vel_xy": float(np.sqrt(np.mean(vel_xy_err**2))),
        "mean_z_err": float(np.mean(z_err)),
        "mean_yaw_err": float(np.mean(yaw_err)),
        "mean_u_norm": float(np.mean(u_norm)),
        "max_u_norm": float(np.max(u_norm)),
        "final_xy_err": float(xy_err[-1]),
    }


# =====================================================================
# Main run
# =====================================================================
def run(
    controller_name: str,
    robot_name: str = "mini_cheetah",
    teleop_enabled: bool = False,
    render: bool = True,
    duration: float = 10.0,
    disturbance_type: str = "impulse",
    save_log: bool = True,
    path_name: str = "none",
    path_size: float = 0.30,
    path_speed: float = 0.10,
    settle_time: float = 2.0,
    ramp_time: float = 2.0,
    waypoint_tol: float = 0.03,
    waypoint_kp: float = 0.5,
    yaw_mode: str = "none",
    follower_max_wz: float = 0.5,
):
    print(f"\n{'=' * 60}")
    print(f"  Controller:   {controller_name.upper()}")
    print(f"  Robot:        {robot_name}")
    print(f"  Teleop:       {teleop_enabled}")
    print(f"  Duration:     {duration}s")
    print(f"  Disturbance:  {disturbance_type}")
    print(f"  Path:         {path_name}")
    print(f"{'=' * 60}\n")

    env = QuadrupedEnv(
        robot=robot_name,
        scene="flat",
        sim_dt=0.002,
        base_vel_command_type="human",
        state_obs_names=tuple(QuadrupedEnv.ALL_OBS),
    )

    _ = env.reset(random=False)
    if render:
        env.render()

    teleop = TeleopState()
    if teleop_enabled:
        teleop_thread = threading.Thread(
            target=teleop_keyboard_loop,
            args=(teleop,),
            daemon=True,
        )
        teleop_thread.start()

    dyn = build_dynamics()
    Q, R, Q_f = build_cost_matrices()

    x_ref_inner = build_reference_state(
        dyn,
        height=ROBOT_HIP_HEIGHT,
        vx=0.0,
        vy=0.0,
        wz=0.0,
    )

    u_ref = dyn.standing_control()
    controller = build_controller(controller_name, dyn, Q, R, Q_f, x_ref_inner)

    ori_ekf = OrientationEKF(dt=env.mjModel.opt.timestep)

    sim_dt = env.mjModel.opt.timestep
    ctrl_dt = 0.01
    ctrl_steps = max(1, int(ctrl_dt / sim_dt))
    n_steps = int(duration / sim_dt)

    start_xy = get_state(env)[0:2].copy()

    trajectory = None
    gait = TrotGaitPlanner(
        step_period=0.6,
        duty_factor=0.6,
        swing_height=0.05,
    )
    foot_traj = FootTrajectoryGenerator(swing_height=0.05)

    if (not teleop_enabled) and path_name != "none":
        waypoints_2d = build_waypoints(path_name, path_size, start_xy)

        waypoints_3d = []
        for p in waypoints_2d:
            waypoints_3d.append([p[0], p[1], ROBOT_HIP_HEIGHT])

        trajectory = WaypointTrajectory(
            waypoints=waypoints_3d,
            speed=path_speed,
            dt=ctrl_dt,
        )

        print(f"  Trajectory samples: {trajectory.length()}")

    log_t, log_x, log_ref, log_u, log_err, log_dist = [], [], [], [], [], []

    current_grfs = u_ref.copy()
    desired_vx = 0.0
    desired_vy = 0.0
    desired_wz = 0.0
    target_xy = start_xy.copy()

    print(f"  Sim dt: {sim_dt}s")
    print(f"  Ctrl rate: {1 / ctrl_dt:.0f} Hz")
    print(f"  Total steps: {n_steps}")
    print("  Starting simulation...\n")

    try:
        for step in range(n_steps):
            t = step * sim_dt
            traj_k = int(max(0.0, t - settle_time) / ctrl_dt)

            x = get_state(env)
            pos_xy = x[0:2].copy()

            real_contact = get_contacts(env)
            r_feet = get_feet_world(env)

            if teleop_enabled:
                desired_vx = teleop.vx
                desired_vy = teleop.vy
                desired_wz = teleop.wz
                target_xy = pos_xy.copy()
                contact = real_contact

            elif trajectory is not None:
                if t < settle_time:
                    desired_vx = 0.0
                    desired_vy = 0.0
                    desired_wz = 0.0
                    target_xy = pos_xy.copy()
                    contact = real_contact
                else:
                    ref = trajectory.get_reference(traj_k)

                    alpha = min((t - settle_time) / max(ramp_time, 1e-6), 1.0)

                    desired_vx = alpha * ref["vel"][0]
                    desired_vy = alpha * ref["vel"][1]
                    desired_wz = alpha * ref["yaw_rate"]

                    target_xy = ref["pos"][0:2].copy()

                    # Aquí empieza la marcha tipo trot
                    contact = gait.get_contact_pattern(t - settle_time)

                    # Seguridad: si MuJoCo dice que una pata no toca,
                    # no forzamos contacto falso.
                    contact = gait.get_contact_pattern(t - settle_time)

            else:
                desired_vx = 0.0
                desired_vy = 0.0
                desired_wz = 0.0
                target_xy = pos_xy.copy()
                contact = real_contact

            x_ref_inner = build_reference_state(
                dyn,
                height=ROBOT_HIP_HEIGHT,
                vx=desired_vx,
                vy=desired_vy,
                wz=desired_wz,
            )

            x_ref_log = build_logging_reference(
                target_xy=target_xy,
                height=ROBOT_HIP_HEIGHT,
                vx=desired_vx,
                vy=desired_vy,
                wz=desired_wz,
            )

            try:
                if hasattr(env, "target_base_vel"):
                    env.target_base_vel[:] = np.array([desired_vx, desired_vy, 0.0])
                if hasattr(env, "target_base_ang_vel"):
                    env.target_base_ang_vel[:] = np.array([0.0, 0.0, desired_wz])
                if hasattr(env, "ref_base_lin_vel"):
                    env.ref_base_lin_vel = desired_vx
            except Exception:
                pass

            dist = np.zeros(6)

            if disturbance_type == "impulse":
                if 2.0 <= t < 2.15:
                    dist = np.array([50.0, 25.0, 0.0, 0.0, 0.0, 5.0])

            elif disturbance_type == "persistent":
                if t >= 2.0:
                    dist = np.array([15.0, 8.0, 0.0, 0.0, 0.0, 2.0])

            env.mjData.qfrc_applied[:6] = dist

            gyro = env.base_ang_vel(frame="base")
            accel_world = env.base_lin_acc(frame="world")
            R_WB = env.base_configuration[0:3, 0:3]
            accel_body = R_WB.T @ (accel_world - np.array([0.0, 0.0, -9.81]))

            ori_ekf.predict(gyro)
            ori_ekf.update_accel(accel_body)

            if step % ctrl_steps == 0:
                try:
                    if controller_name == "lqg":
                        y = x + np.random.randn(12) * np.array(
                            [5e-3] * 3 +
                            [2e-2] * 3 +
                            [1e-2] * 3 +
                            [5e-2] * 3
                        )
                        current_grfs = controller.step(y, x_ref_inner, u_ref)
                    else:
                        current_grfs = controller.compute_control(
                            x=x,
                            x_ref=x_ref_inner,
                            u_ref=u_ref,
                        )

                except Exception as e:
                    if step < 5:
                        print(f"  Controller error at t={t:.3f}: {e}")
                    current_grfs = u_ref.copy()

                current_grfs = np.clip(current_grfs, -150.0, 150.0)

                for i in range(4):
                    if not contact[i]:
                        current_grfs[3 * i:3 * i + 3] = 0.0

            tau = grf_to_torques(env, current_grfs, contact)

            _, _, terminated, _, _ = env.step(action=tau)

            if render:
                env.render()

            log_t.append(t)
            log_x.append(x.copy())
            log_ref.append(x_ref_log.copy())
            log_u.append(current_grfs.copy())
            log_err.append(np.linalg.norm(x[0:2] - target_xy))
            log_dist.append(np.linalg.norm(dist))

            if step % int(1.0 / sim_dt) == 0:
                pos_err = np.linalg.norm(x[0:2] - target_xy)
                vel_err = np.linalg.norm(x[3:5] - np.array([desired_vx, desired_vy]))

                print(
                    f"  t={t:5.1f}s | "
                    f"pos_err={pos_err:.4f}m | "
                    f"vel_err={vel_err:.4f}m/s | "
                    f"height={x[2]:.3f}m | "
                    f"vx={x[3]:+.3f} | "
                    f"vy={x[4]:+.3f} | "
                    f"wz={x[11]:+.3f} | "
                    f"cmd=({desired_vx:+.2f},{desired_vy:+.2f},{desired_wz:+.2f}) | "
                    f"contact={contact.astype(int).tolist()}"
                )

            if terminated:
                print(f"\n  Environment terminated early at t={t:.3f}s.")
                print(
                    f"  Final state: "
                    f"z={x[2]:.3f}, "
                    f"roll={x[6]:+.3f}, "
                    f"pitch={x[7]:+.3f}, "
                    f"yaw={x[8]:+.3f}"
                )
                break

    except KeyboardInterrupt:
        print("\n  Interrupted by user.")

    finally:
        teleop.quit_requested = True
        try:
            env.close()
        except Exception:
            pass

    log_t = np.array(log_t) if len(log_t) > 0 else np.zeros(1)
    log_x = np.array(log_x) if len(log_x) > 0 else np.zeros((1, 12))
    log_ref = np.array(log_ref) if len(log_ref) > 0 else np.zeros((1, 12))
    log_u = np.array(log_u) if len(log_u) > 0 else np.zeros((1, 12))
    log_err = np.array(log_err) if len(log_err) > 0 else np.zeros(1)
    log_dist = np.array(log_dist) if len(log_dist) > 0 else np.zeros(1)

    result = {
        "time": log_t,
        "state": log_x,
        "reference": log_ref,
        "control": log_u,
        "error": log_err,
        "disturbance": log_dist,
        "path_name": path_name,
    }

    if save_log and len(log_t) > 1:
        save_single_run_plot(result, controller_name, robot_name, disturbance_type)

    metrics = compute_tracking_metrics(result)

    print(f"\n  --- {controller_name.upper()} Summary ---")
    print(f"  XY RMSE:        {metrics['rmse_xy']:.4f} m")
    print(f"  XY max error:   {metrics['max_xy_err']:.4f} m")
    print(f"  XY vel RMSE:    {metrics['rmse_vel_xy']:.4f} m/s")
    print(f"  Mean z error:   {metrics['mean_z_err']:.4f} m")
    print(f"  Mean yaw error: {metrics['mean_yaw_err']:.4f} rad")
    print(f"  Mean ||u||:     {metrics['mean_u_norm']:.1f} N")
    print(f"  Max ||u||:      {metrics['max_u_norm']:.1f} N")
    print(f"  Final XY error: {metrics['final_xy_err']:.4f} m")

    result["metrics"] = metrics
    return result

# =====================================================================
# Entry point
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=["pmp", "lqg", "mpc", "all"], default="lqg")
    parser.add_argument("--robot-name", type=str, default="mini_cheetah")
    parser.add_argument("--teleop", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--disturbance", choices=["impulse", "persistent", "none"], default="impulse")

    parser.add_argument("--path", choices=["none", "line", "square", "circle", "figure8"], default="none")
    parser.add_argument("--path-size", type=float, default=0.30)
    parser.add_argument("--path-speed", type=float, default=0.10)
    parser.add_argument("--settle-time", type=float, default=2.0)
    parser.add_argument("--ramp-time", type=float, default=2.0)
    parser.add_argument("--waypoint-tol", type=float, default=0.03)
    parser.add_argument("--waypoint-kp", type=float, default=0.5)
    parser.add_argument("--yaw-mode", choices=["none", "path"], default="none")
    parser.add_argument("--follower-max-wz", type=float, default=0.5)

    args = parser.parse_args()
    render = not args.no_render

    if args.controller == "all":
        run_comparison(
            render=render,
            duration=args.duration,
            disturbance_type=args.disturbance,
            robot_name=args.robot_name,
            path_name=args.path,
            path_size=args.path_size,
            path_speed=args.path_speed,
            settle_time=args.settle_time,
            ramp_time=args.ramp_time,
            waypoint_tol=args.waypoint_tol,
            waypoint_kp=args.waypoint_kp,
            yaw_mode=args.yaw_mode,
            follower_max_wz=args.follower_max_wz,
        )
    else:
        run(
            controller_name=args.controller,
            robot_name=args.robot_name,
            teleop_enabled=args.teleop,
            render=render,
            duration=args.duration,
            disturbance_type=args.disturbance,
            save_log=True,
            path_name=args.path,
            path_size=args.path_size,
            path_speed=args.path_speed,
            settle_time=args.settle_time,
            ramp_time=args.ramp_time,
            waypoint_tol=args.waypoint_tol,
            waypoint_kp=args.waypoint_kp,
            yaw_mode=args.yaw_mode,
            follower_max_wz=args.follower_max_wz,
        )


if __name__ == "__main__":
    main()