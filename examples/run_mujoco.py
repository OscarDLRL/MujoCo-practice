#!/usr/bin/env python3
"""Run PMP / LQG / MPC controllers on a quadruped robot in MuJoCo with real
waypoint tracking, automatic metrics, and controller comparison.

Examples:
    python examples/run_mujoco.py --controller lqg --path line --duration 10
    python examples/run_mujoco.py --controller mpc --path square --duration 16
    python examples/run_mujoco.py --controller all --path zigzag --duration 18 --no-render
    python examples/run_mujoco.py --controller lqg --teleop
"""

import sys
import os
import csv
import argparse
import threading
import select
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.utils.quadruped_utils import LegsAttr

from src.dynamics import QuadrupedDynamics
from src.estimator_ekf import OrientationEKF
from src.controller_pmp import PontryaginController
from src.controller_lqg import LQGController
from src.controller_mpc import MPCController
from src.trajectory_generator import WaypointTrajectory
from quadruped_pympc.quadruped_pympc_wrapper import QuadrupedPyMPC_Wrapper
from quadruped_pympc import config as qpympc_cfg
from gym_quadruped.robot_cfgs import get_robot_config


# =====================================================================
# Nominal physical parameters
# =====================================================================
ROBOT_MASS = 9.0
ROBOT_BODY_MASS = 6.921
ROBOT_INERTIA = np.diag([0.107, 0.098, 0.024])
ROBOT_HIP_HEIGHT = 0.225
ROBOT_FOOT_OFFSET = np.array([
    [0.19, 0.111, -0.225],
    [0.19, -0.111, -0.225],
    [-0.19, 0.111, -0.225],
    [-0.19, -0.111, -0.225],
])


def configure_qpympc_for_robot(robot_name: str):
    rc = get_robot_config(robot_name=robot_name)

    qpympc_cfg.robot = robot_name
    qpympc_cfg.robot_cfg = rc
    qpympc_cfg.robot_leg_joints = rc.leg_joints
    qpympc_cfg.robot_feet_geom_names = rc.feet_geom_names
    qpympc_cfg.qpos0_js = rc.qpos0_js
    qpympc_cfg.hip_height = rc.hip_height

    if robot_name == "mini_cheetah":
        qpympc_cfg.mass = 12.5
        qpympc_cfg.inertia = np.array([
            [1.58460467e-01, 1.21660000e-04, -1.55444692e-02],
            [1.21660000e-04, 4.68645637e-01, -3.12000000e-05],
            [-1.55444692e-02, -3.12000000e-05, 5.24474661e-01]
        ])

    # tuning estabilidad primero, velocidad después
    qpympc_cfg.simulation_params["gait"] = "trot"
    qpympc_cfg.simulation_params["gait_params"]["trot"]["step_freq"] = 1.15
    qpympc_cfg.simulation_params["gait_params"]["trot"]["duty_factor"] = 0.72

    qpympc_cfg.simulation_params["step_height"] = 0.12 * rc.hip_height
    qpympc_cfg.simulation_params["swing_position_gain_fb"] = 400
    qpympc_cfg.simulation_params["swing_velocity_gain_fb"] = 8

    qpympc_cfg.simulation_params["impedence_joint_position_gain"] = 16.0
    qpympc_cfg.simulation_params["impedence_joint_velocity_gain"] = 3.0

    qpympc_cfg.simulation_params["ref_z"] = rc.hip_height


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
                        if seq2 == "A":
                            teleop.vx += teleop.step_lin
                        elif seq2 == "B":
                            teleop.vx -= teleop.step_lin
                        elif seq2 == "C":
                            teleop.wz -= teleop.step_ang
                        elif seq2 == "D":
                            teleop.wz += teleop.step_ang
                elif ch == "z":
                    teleop.vy += teleop.step_lin
                elif ch == "c":
                    teleop.vy -= teleop.step_lin
                elif ch == " ":
                    teleop.zero()

                teleop.clamp()
                print(
                    f"\rcmd -> vx={teleop.vx:+.2f}, vy={teleop.vy:+.2f}, wz={teleop.wz:+.2f}   ",
                    end="",
                    flush=True,
                )
    except KeyboardInterrupt:
        teleop.quit_requested = True
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print()


# =====================================================================
# Helpers
# =====================================================================
def wrap_to_pi(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def get_state(env) -> np.ndarray:
    p = env.base_pos.copy()
    v = env.base_lin_vel(frame="world")
    rpy = env.base_ori_euler_xyz.copy()
    omega = env.base_ang_vel(frame="base")
    return np.concatenate([p, v, rpy, omega])


def get_contacts(env) -> np.ndarray:
    try:
        cs, _ = env.feet_contact_state()
        return np.array([cs.FL, cs.FR, cs.RL, cs.RR], dtype=bool)
    except Exception:
        return np.ones(4, dtype=bool)


def get_feet_world(env) -> np.ndarray | None:
    try:
        fp = env.feet_pos(frame="world")
        return np.array([fp.FL, fp.FR, fp.RL, fp.RR])
    except Exception:
        return None


def ensure_results_dir() -> None:
    os.makedirs("results", exist_ok=True)


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
        80, 80, 400,
        8, 8, 40,
        150, 150, 30,
        1, 1, 4,
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
    x_ref = dyn.standing_state(height=height)
    x_ref[3:6] = np.array([vx, vy, 0.0])
    x_ref[6:9] = np.array([0.0, 0.0, 0.0])
    x_ref[9:12] = np.array([0.0, 0.0, wz])
    return x_ref



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
# Paths and path follower
# =====================================================================
def make_relative_waypoints(path_name: str, z_ref: float, side: float = 1.0):
    z = z_ref
    if path_name == "line":
        return [
            [0.0, 0.0, z],
            [1.5, 0.0, z],
        ]
    if path_name == "square":
        return [
            [0.0, 0.0, z],
            [side, 0.0, z],
            [side, side, z],
            [0.0, side, z],
            [0.0, 0.0, z],
        ]
    if path_name == "zigzag":
        return [
            [0.0, 0.0, z],
            [0.5, 0.35, z],
            [1.0, -0.35, z],
            [1.5, 0.35, z],
            [2.0, 0.0, z],
        ]
    raise ValueError(f"Unknown path: {path_name}")



def shift_waypoints_to_world(relative_waypoints, start_pos: np.ndarray, z_ref: float):
    rel = np.array(relative_waypoints, dtype=float)
    rel[:, 0] += start_pos[0]
    rel[:, 1] += start_pos[1]
    rel[:, 2] = z_ref
    return rel.tolist()



def waypoint_follower(
    x: np.ndarray,
    ref: dict,
    kp_xy: float = 0.9,
    kd_xy: float = 0.12,
    kp_yaw: float = 1.3,
    vmax: float = 0.40,
    vymax: float = 0.25,
    wzmax: float = 1.0,
):
    pos_err_xy = ref["pos"][:2] - x[:2]
    vel_ff_xy = ref["vel"][:2]
    vel_fb_xy = kp_xy * pos_err_xy - kd_xy * x[3:5]

    cmd_xy = vel_ff_xy + vel_fb_xy
    cmd_xy[0] = np.clip(cmd_xy[0], -vmax, vmax)
    cmd_xy[1] = np.clip(cmd_xy[1], -vymax, vymax)

    yaw_ref = ref["yaw"]
    yaw_err = wrap_to_pi(yaw_ref - x[8])
    cmd_wz = float(np.clip(kp_yaw * yaw_err, -wzmax, wzmax))

    return float(cmd_xy[0]), float(cmd_xy[1]), cmd_wz, pos_err_xy, yaw_err



def controller_velocity_correction(
    grfs: np.ndarray,
    feet_world: np.ndarray | None,
    com_pos: np.ndarray,
    dt: float,
):
    grf_mat = grfs.reshape(4, 3)
    f_total = np.sum(grf_mat, axis=0)

    dv_xy = np.clip((f_total[:2] / ROBOT_MASS) * dt, -0.015, 0.015)
    dwz = 0.0

    if feet_world is not None:
        tau_total = np.zeros(3)
        for i in range(4):
            r = feet_world[i] - com_pos
            tau_total += np.cross(r, grf_mat[i])
        iz = float(ROBOT_INERTIA[2, 2])
        dwz = float(np.clip((tau_total[2] / max(iz, 1e-6)) * dt, -0.20, 0.20))

    return dv_xy, dwz


# =====================================================================
# Logging and plotting
# =====================================================================
def compute_path_length(waypoints_world: np.ndarray) -> float:
    if len(waypoints_world) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(waypoints_world[:, :2], axis=0), axis=1)))



def nearest_waypoint_index(pos_xy: np.ndarray, traj_samples) -> int:
    pts = np.array([s["pos"][:2] for s in traj_samples])
    d = np.linalg.norm(pts - pos_xy[None, :], axis=1)
    return int(np.argmin(d))



def compute_metrics(result: dict, traj: WaypointTrajectory, waypoints_world: np.ndarray, duration: float):
    time = result["time"]
    state = result["state"]
    ref = result["ref"]
    cmd = result["cmd"]

    if len(time) == 0:
        return {
            "tracking_rmse_xy": np.nan,
            "tracking_rmse_xyz": np.nan,
            "mean_vel_err": np.nan,
            "final_waypoint_error": np.nan,
            "path_completion_pct": 0.0,
            "survival_time": 0.0,
            "mean_cmd_speed": np.nan,
            "mean_height": np.nan,
            "termination": "empty",
            "path_length": compute_path_length(waypoints_world),
            "distance_travelled": 0.0,
        }

    pos_xy_err = np.linalg.norm(state[:, :2] - ref[:, :2], axis=1)
    pos_xyz_err = np.linalg.norm(state[:, :3] - ref[:, :3], axis=1)
    vel_err = np.linalg.norm(state[:, 3:6] - ref[:, 3:6], axis=1)
    final_waypoint_error = float(np.linalg.norm(state[-1, :2] - waypoints_world[-1, :2]))

    traj_samples = traj.samples
    last_nearest = nearest_waypoint_index(state[-1, :2], traj_samples)
    completion_pct = 100.0 * last_nearest / max(1, len(traj_samples) - 1)

    distance_travelled = 0.0
    if len(state) > 1:
        distance_travelled = float(np.sum(np.linalg.norm(np.diff(state[:, :2], axis=0), axis=1)))

    path_length = compute_path_length(waypoints_world)

    metrics = {
        "tracking_rmse_xy": float(np.sqrt(np.mean(pos_xy_err ** 2))),
        "tracking_rmse_xyz": float(np.sqrt(np.mean(pos_xyz_err ** 2))),
        "mean_vel_err": float(np.mean(vel_err)),
        "final_waypoint_error": final_waypoint_error,
        "path_completion_pct": float(np.clip(completion_pct, 0.0, 100.0)),
        "survival_time": float(time[-1]),
        "mean_cmd_speed": float(np.mean(np.linalg.norm(cmd[:, :2], axis=1))),
        "mean_height": float(np.mean(state[:, 2])),
        "termination": result.get("termination_reason", "unknown"),
        "path_length": path_length,
        "distance_travelled": distance_travelled,
        "full_duration_completed": bool(time[-1] >= max(0.0, duration - 2e-3)),
    }
    return metrics



def append_metrics_csv(metrics_path: str, row: dict):
    ensure_results_dir()
    exists = os.path.exists(metrics_path)
    with open(metrics_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)



def save_single_run_plot(result, metrics, controller_name, robot_name, disturbance_type, path_name, waypoints_world):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_results_dir()

    t = result["time"]
    x = result["state"]
    u = result["control"]
    ref = result["ref"]
    cmd = result["cmd"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"{controller_name.upper()} | {robot_name} | path={path_name} | disturbance={disturbance_type}",
        fontsize=13,
        fontweight="bold",
    )

    # XY path
    axes[0, 0].plot(ref[:, 0], ref[:, 1], "--", label="reference path")
    axes[0, 0].plot(x[:, 0], x[:, 1], label="robot path")
    axes[0, 0].scatter(waypoints_world[:, 0], waypoints_world[:, 1], marker="o", s=45, label="waypoints")
    axes[0, 0].set_title("XY trajectory")
    axes[0, 0].set_xlabel("x [m]")
    axes[0, 0].set_ylabel("y [m]")
    axes[0, 0].axis("equal")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)

    # Tracking errors
    pos_xy_err = np.linalg.norm(x[:, :2] - ref[:, :2], axis=1)
    vel_err = np.linalg.norm(x[:, 3:6] - ref[:, 3:6], axis=1)
    axes[0, 1].plot(t, pos_xy_err, label="XY error [m]")
    axes[0, 1].plot(t, vel_err, label="velocity error [m/s]")
    axes[0, 1].set_title("Tracking errors")
    axes[0, 1].set_xlabel("time [s]")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=8)

    # Commands vs actual
    axes[1, 0].plot(t, x[:, 3], label="vx")
    axes[1, 0].plot(t, x[:, 4], label="vy")
    axes[1, 0].plot(t, cmd[:, 0], "--", label="cmd vx")
    axes[1, 0].plot(t, cmd[:, 1], "--", label="cmd vy")
    axes[1, 0].set_title("Planar velocities")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("m/s")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=8)

    # Height + GRF norm
    axes2 = axes[1, 1]
    axes2.plot(t, x[:, 2], label="base height [m]")
    axes2.plot(t, np.linalg.norm(u, axis=1) / 200.0, label="||GRF|| / 200")
    axes2.set_title("Height and control effort")
    axes2.set_xlabel("time [s]")
    axes2.grid(True, alpha=0.3)
    axes2.legend(fontsize=8)

    txt = (
        f"RMSE xy: {metrics['tracking_rmse_xy']:.3f} m\n"
        f"Final wp err: {metrics['final_waypoint_error']:.3f} m\n"
        f"Completion: {metrics['path_completion_pct']:.1f} %\n"
        f"Survival: {metrics['survival_time']:.2f} s\n"
        f"Termination: {metrics['termination']}"
    )
    fig.text(0.73, 0.08, txt, fontsize=9,
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    plt.tight_layout()
    out_path = f"results/mujoco_{controller_name}_{robot_name}_{path_name}_{disturbance_type}.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"\n  Plot saved: {out_path}")



def save_comparison_plot(results, comparison_rows, robot_name, disturbance_type, path_name, waypoints_world):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ensure_results_dir()

    colors = {"pmp": "tab:red", "lqg": "tab:green", "mpc": "tab:blue"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Controller comparison | {robot_name} | path={path_name} | disturbance={disturbance_type}",
        fontsize=13,
        fontweight="bold",
    )

    # XY trajectories
    axes[0, 0].scatter(waypoints_world[:, 0], waypoints_world[:, 1], s=40, label="waypoints")
    for name, data in results.items():
        axes[0, 0].plot(data["state"][:, 0], data["state"][:, 1], label=name.upper(), color=colors[name])
    axes[0, 0].set_title("XY trajectories")
    axes[0, 0].axis("equal")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)

    # XY error over time
    for name, data in results.items():
        xy_err = np.linalg.norm(data["state"][:, :2] - data["ref"][:, :2], axis=1)
        axes[0, 1].plot(data["time"], xy_err, label=name.upper(), color=colors[name])
    axes[0, 1].set_title("XY tracking error")
    axes[0, 1].set_xlabel("time [s]")
    axes[0, 1].set_ylabel("m")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=8)

    labels = [r["controller"].upper() for r in comparison_rows]
    rmse = [r["tracking_rmse_xy"] for r in comparison_rows]
    completion = [r["path_completion_pct"] for r in comparison_rows]

    axes[1, 0].bar(labels, rmse)
    axes[1, 0].set_title("RMSE xy")
    axes[1, 0].set_ylabel("m")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    axes[1, 1].bar(labels, completion)
    axes[1, 1].set_title("Path completion")
    axes[1, 1].set_ylabel("%")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = f"results/mujoco_comparison_{robot_name}_{path_name}_{disturbance_type}.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"\n  Comparison plot saved: {out_path}")


# =====================================================================
# Main run
# =====================================================================
def run(
    controller_name: str,
    robot_name: str = "mini_cheetah",
    teleop_enabled: bool = False,
    render: bool = True,
    duration: float = 10.0,
    disturbance_type: str = "none",
    save_log: bool = True,
    path_name: str = "line",
    path_speed: float = 0.15,
):
    print(f"\n{'=' * 60}")
    print(f"  Controller:   {controller_name.upper()}")
    print(f"  Robot:        {robot_name}")
    print(f"  Path:         {path_name}")
    print(f"  Teleop:       {teleop_enabled}")
    print(f"  Duration:     {duration}s")
    print(f"  Disturbance:  {disturbance_type}")
    print(f"{'=' * 60}\n")

    ensure_results_dir()

    state_obs_names = tuple(QuadrupedEnv.ALL_OBS)
    env = QuadrupedEnv(
        robot=robot_name,
        scene="flat",
        sim_dt=0.002,
        base_vel_command_type="human",
        state_obs_names=state_obs_names,
    )

    _ = env.reset(random=False)
    nominal_base_height = float(env.base_pos[2])
    print(f"  Nominal base height detected: {nominal_base_height:.3f} m")
    if render:
        env.render()

    teleop = TeleopState()
    teleop_thread = None
    if teleop_enabled:
        teleop_thread = threading.Thread(
            target=teleop_keyboard_loop,
            args=(teleop,),
            daemon=True,
        )
        teleop_thread.start()

    dyn = build_dynamics()
    Q, R, Q_f = build_cost_matrices()

    x0 = get_state(env)
    world_waypoints = shift_waypoints_to_world(
        make_relative_waypoints(path_name, nominal_base_height),
        x0[:3],
        nominal_base_height,
    )
    traj = WaypointTrajectory(world_waypoints, speed=path_speed, dt=env.mjModel.opt.timestep)
    waypoints_world_np = np.array(world_waypoints)

    x_ref = build_reference_state(dyn, height=nominal_base_height, vx=0.0, vy=0.0, wz=0.0)
    u_ref = dyn.standing_control()
    controller = build_controller(controller_name, dyn, Q, R, Q_f, x_ref)

    ori_ekf = OrientationEKF(dt=env.mjModel.opt.timestep)

    sim_dt = env.mjModel.opt.timestep
    ctrl_dt = 0.01
    ctrl_steps = max(1, int(ctrl_dt / sim_dt))
    if duration is None:
        # usar longitud del path / velocidad deseada
        path_length = compute_path_length(waypoints_world_np)

        # tiempo nominal para recorrer trayectoria
        duration = path_length / path_speed

        # margen para aceleración, settling, maniobras
        duration *= 1.4

        print(f"Auto duration selected: {duration:.2f} s")

    n_steps = int(duration / sim_dt)

    log_t = []
    log_x = []
    log_u = []
    log_ref = []
    log_cmd = []
    log_dist = []
    current_grfs = u_ref.copy()
    cmd_filtered = np.zeros(3)
    termination_reason = "completed"

    print(f"  Sim dt: {sim_dt}s, Ctrl rate: {1 / ctrl_dt:.0f} Hz, Total steps: {n_steps}")
    print("  Starting simulation...\n")

    try:
        configure_qpympc_for_robot(robot_name)
        pympc = QuadrupedPyMPC_Wrapper(
            initial_feet_pos=env.feet_pos,
            legs_order=("FL", "FR", "RL", "RR"),
        )

        for step in range(n_steps):
            t = step * sim_dt

            x = get_state(env)
            contact = get_contacts(env)
            r_feet = get_feet_world(env)

            ref = traj.get_reference(step)

            if teleop_enabled:
                cmd_vx, cmd_vy, cmd_wz = teleop.vx, teleop.vy, teleop.wz
                pos_err_xy = ref["pos"][:2] - x[:2]
                yaw_err = wrap_to_pi(ref["yaw"] - x[8])
            else:
                cmd_vx, cmd_vy, cmd_wz, pos_err_xy, yaw_err = waypoint_follower(
                    x,
                    ref,
                    kp_xy=0.45,
                    kd_xy=0.10,
                    kp_yaw=0.8,
                    vmax=0.18,
                    vymax=0.08,
                    wzmax=0.45,
                )

            cmd_target = np.array([cmd_vx, cmd_vy, cmd_wz])
            alpha = 0.08
            cmd_filtered = (1.0 - alpha) * cmd_filtered + alpha * cmd_target
            cmd_vx = float(cmd_filtered[0])
            cmd_vy = float(cmd_filtered[1])
            cmd_wz = float(cmd_filtered[2])

            if t < 0.8:
                cmd_vx = 0.0
                cmd_vy = 0.0
                cmd_wz = 0.0

            x_ref = build_reference_state(
                dyn,
                height=nominal_base_height,
                vx=cmd_vx,
                vy=cmd_vy,
                wz=cmd_wz,
            )
            x_ref[0:3] = ref["pos"]
            x_ref[6:9] = np.array([0.0, 0.0, ref["yaw"]])

            try:
                if hasattr(env, "target_base_vel"):
                    env.target_base_vel[:] = np.array([cmd_vx, cmd_vy, 0.0])
                if hasattr(env, "target_base_ang_vel"):
                    env.target_base_ang_vel[:] = np.array([0.0, 0.0, cmd_wz])
            except Exception:
                pass

            dist = np.zeros(6)
            if disturbance_type == "impulse":
                if 2.0 <= t < 2.15:
                    dist = np.array([35.0, 15.0, 0.0, 0.0, 0.0, 3.0])
            elif disturbance_type == "persistent":
                if t >= 2.0:
                    dist = np.array([10.0, 5.0, 0.0, 0.0, 0.0, 1.5])
            env.mjData.qfrc_applied[:6] = dist

            gyro = env.base_ang_vel(frame="base")
            accel_world = env.base_lin_acc(frame="world")
            R_WB = env.base_configuration[0:3, 0:3]
            accel_body = R_WB.T @ (accel_world - np.array([0.0, 0.0, -9.81]))
            ori_ekf.predict(gyro)
            ori_ekf.update_accel(accel_body)

            if step % ctrl_steps == 0:
                try:
                    _A_c_new, _B_c_new = dyn.continuous_AB(x, contact, r_feet)
                    _A_d_new, _B_d_new = dyn.discretize(_A_c_new, _B_c_new)
                    _g_d = dyn.gravity_vector()
                    _ = (_A_d_new, _B_d_new, _g_d)
                except Exception:
                    pass

                try:
                    if controller_name == "lqg":
                        y = x + np.random.randn(12) * np.array(
                            [5e-3] * 3 + [2e-2] * 3 + [1e-2] * 3 + [5e-2] * 3
                        )
                        current_grfs = controller.step(y, x_ref, u_ref)
                    else:
                        current_grfs = controller.compute_control(
                            x=x,
                            x_ref=x_ref,
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

                dv_xy, dwz = controller_velocity_correction(
                    current_grfs,
                    r_feet,
                    env.com.copy(),
                    ctrl_dt,
                )
                if not teleop_enabled:
                    cmd_vx = float(np.clip(cmd_vx + dv_xy[0], -0.45, 0.45))
                    cmd_vy = float(np.clip(cmd_vy + dv_xy[1], -0.30, 0.30))
                    cmd_wz = float(np.clip(cmd_wz + 0.25 * dwz, -0.5, 0.5))

            feet_pos = env.feet_pos(frame="world")
            feet_vel = env.feet_vel(frame="world")
            hip_pos = env.hip_positions(frame="world")

            base_lin_vel = env.base_lin_vel(frame="world")
            base_ang_vel = env.base_ang_vel(frame="base")
            base_ori_euler_xyz = env.base_ori_euler_xyz

            base_pos = env.base_pos.copy()
            com_pos = env.com.copy()

            qpos = env.mjData.qpos
            qvel = env.mjData.qvel

            legs_qvel_idx = env.legs_qvel_idx
            legs_qpos_idx = env.legs_qpos_idx

            joints_pos = LegsAttr(
                FL=qpos[legs_qpos_idx.FL],
                FR=qpos[legs_qpos_idx.FR],
                RL=qpos[legs_qpos_idx.RL],
                RR=qpos[legs_qpos_idx.RR],
            )

            legs_mass_matrix = env.legs_mass_matrix
            legs_qfrc_bias = env.legs_qfrc_bias
            legs_qfrc_passive = env.legs_qfrc_passive

            feet_jac = env.feet_jacobians(frame="world", return_rot_jac=False)
            feet_jac_dot = env.feet_jacobians_dot(frame="world", return_rot_jac=False)
            inertia = env.get_base_inertia().flatten()

            ref_base_lin_vel = np.array([cmd_vx, cmd_vy, 0.0])
            ref_base_ang_vel = np.array([0.0, 0.0, cmd_wz])

            tau_init = LegsAttr(
                FL=np.zeros(3),
                FR=np.zeros(3),
                RL=np.zeros(3),
                RR=np.zeros(3),
            )

            tau_leg = pympc.compute_actions(
                com_pos,
                base_pos,
                base_lin_vel,
                base_ori_euler_xyz,
                base_ang_vel,
                feet_pos,
                hip_pos,
                joints_pos,
                None,
                ("FL", "FR", "RL", "RR"),
                sim_dt,
                ref_base_lin_vel,
                ref_base_ang_vel,
                env.step_num,
                qpos,
                qvel,
                feet_jac,
                feet_jac_dot,
                feet_vel,
                legs_qfrc_passive,
                legs_qfrc_bias,
                legs_mass_matrix,
                legs_qpos_idx,
                legs_qvel_idx,
                tau_init,
                inertia,
                env.mjData.contact,
            )

            action = np.zeros(env.mjModel.nu)
            action[env.legs_tau_idx.FL] = tau_leg.FL
            action[env.legs_tau_idx.FR] = tau_leg.FR
            action[env.legs_tau_idx.RL] = tau_leg.RL
            action[env.legs_tau_idx.RR] = tau_leg.RR

            _, _, terminated, _, _ = env.step(action=action)

            if render:
                env.render()

            log_t.append(t)
            log_x.append(x.copy())
            log_u.append(current_grfs.copy())
            log_ref.append(x_ref.copy())
            log_cmd.append(np.array([cmd_vx, cmd_vy, cmd_wz]))
            log_dist.append(np.linalg.norm(dist))

            if step % int(1.0 / sim_dt) == 0:
                pos_err = np.linalg.norm(x[:2] - ref["pos"][:2])
                vel_err = np.linalg.norm(x[3:6] - x_ref[3:6])
                print(
                    f"  t={t:5.1f}s | pos_err_xy={pos_err:.4f}m | "
                    f"vel_err={vel_err:.4f}m/s | "
                    f"height={x[2]:.3f}m | "
                    f"yaw_err={yaw_err:+.3f}rad | "
                    f"vx={x[3]:+.3f} | vy={x[4]:+.3f} | wz={x[11]:+.3f} | "
                    f"cmd=({cmd_vx:+.2f},{cmd_vy:+.2f},{cmd_wz:+.2f})"
                )

            if terminated:
                termination_reason = "env_terminated"
                print(f"  Terminated at t={t:.2f}s")
                break

    except KeyboardInterrupt:
        termination_reason = "user_interrupt"
        print("\n  Interrupted by user.")

    finally:
        teleop.quit_requested = True
        env.close()

    if teleop_thread is not None:
        teleop_thread.join(timeout=0.2)

    log_t = np.array(log_t) if len(log_t) > 0 else np.zeros(0)
    log_x = np.array(log_x) if len(log_x) > 0 else np.zeros((0, 12))
    log_u = np.array(log_u) if len(log_u) > 0 else np.zeros((0, 12))
    log_ref = np.array(log_ref) if len(log_ref) > 0 else np.zeros((0, 12))
    log_cmd = np.array(log_cmd) if len(log_cmd) > 0 else np.zeros((0, 3))
    log_dist = np.array(log_dist) if len(log_dist) > 0 else np.zeros(0)

    result = {
        "time": log_t,
        "state": log_x,
        "control": log_u,
        "ref": log_ref,
        "cmd": log_cmd,
        "disturbance": log_dist,
        "termination_reason": termination_reason,
    }

    metrics = compute_metrics(result, traj, waypoints_world_np, duration)

    metrics_row = {
        "controller": controller_name,
        "robot": robot_name,
        "path": path_name,
        "disturbance": disturbance_type,
        **metrics,
    }
    append_metrics_csv("results/metrics_runs.csv", metrics_row)

    if save_log and len(log_t) > 1:
        save_single_run_plot(
            result,
            metrics,
            controller_name,
            robot_name,
            disturbance_type,
            path_name,
            waypoints_world_np,
        )

    print(f"\n  --- {controller_name.upper()} Summary ---")
    print(f"  Tracking RMSE xy:     {metrics['tracking_rmse_xy']:.4f} m")
    print(f"  Tracking RMSE xyz:    {metrics['tracking_rmse_xyz']:.4f} m")
    print(f"  Mean velocity error:  {metrics['mean_vel_err']:.4f} m/s")
    print(f"  Final waypoint error: {metrics['final_waypoint_error']:.4f} m")
    print(f"  Path completion:      {metrics['path_completion_pct']:.1f} %")
    print(f"  Survival time:        {metrics['survival_time']:.2f} s")
    print(f"  Mean GRF norm:        {np.mean(np.linalg.norm(log_u, axis=1)) if len(log_u) else 0.0:.1f} N")
    print(f"  Termination:          {metrics['termination']}")

    return result, metrics, waypoints_world_np


# =====================================================================
# Comparison mode
# =====================================================================
def run_comparison(
    render: bool,
    duration: float,
    disturbance_type: str,
    robot_name: str,
    path_name: str,
    path_speed: float,
):
    results = {}
    comparison_rows = []
    waypoints_world = None

    for name in ["pmp", "lqg", "mpc"]:
        result, metrics, waypoints_world = run(
            name,
            robot_name=robot_name,
            teleop_enabled=False,
            render=render,
            duration=duration,
            disturbance_type=disturbance_type,
            save_log=False,
            path_name=path_name,
            path_speed=path_speed,
        )
        results[name] = result
        comparison_rows.append({
            "controller": name,
            "robot": robot_name,
            "path": path_name,
            "disturbance": disturbance_type,
            **metrics,
        })

    if waypoints_world is not None:
        save_comparison_plot(results, comparison_rows, robot_name, disturbance_type, path_name, waypoints_world)

    cmp_csv = f"results/comparison_{robot_name}_{path_name}_{disturbance_type}.csv"
    with open(cmp_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(f"  Comparison CSV saved: {cmp_csv}")

    print(f"\n{'=' * 72}")
    print(f"  COMPARISON SUMMARY | robot={robot_name} | path={path_name} | dist={disturbance_type}")
    print(f"{'=' * 72}")
    print(f"  {'Ctrl':<8} {'RMSE_xy':>10} {'FinalWP':>10} {'Complete%':>10} {'Survive[s]':>11}")
    print(f"  {'-' * 58}")
    for row in comparison_rows:
        print(
            f"  {row['controller'].upper():<8} "
            f"{row['tracking_rmse_xy']:>10.4f} "
            f"{row['final_waypoint_error']:>10.4f} "
            f"{row['path_completion_pct']:>10.1f} "
            f"{row['survival_time']:>11.2f}"
        )
    print(f"{'=' * 72}")


# =====================================================================
# CLI
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quadruped waypoint tracking with MuJoCo")
    parser.add_argument("--controller", default="lqg", choices=["pmp", "lqg", "mpc", "all"])
    parser.add_argument(
        "--robot-name",
        type=str,
        default="mini_cheetah",
        help="Robot name, e.g. mini_cheetah, aliengo, go2, hyqreal",
    )
    parser.add_argument(
        "--teleop",
        action="store_true",
        help="Enable keyboard teleoperation for commanded velocities",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None
    )
    parser.add_argument("--path", default="line", choices=["line", "square", "zigzag"])
    parser.add_argument("--path-speed", type=float, default=0.15)
    parser.add_argument(
        "--disturbance",
        default="none",
        choices=["impulse", "persistent", "none"],
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Run headless without viewer",
    )
    args = parser.parse_args()

    do_render = not args.no_render

    if args.controller == "all":
        if args.teleop:
            print("Teleop is ignored in comparison mode; running fixed references only.")
        run_comparison(
            render=do_render,
            duration=args.duration,
            disturbance_type=args.disturbance,
            robot_name=args.robot_name,
            path_name=args.path,
            path_speed=args.path_speed,
        )
    else:
        run(
            controller_name=args.controller,
            robot_name=args.robot_name,
            teleop_enabled=args.teleop,
            render=do_render,
            duration=args.duration,
            disturbance_type=args.disturbance,
            save_log=True,
            path_name=args.path,
            path_speed=args.path_speed,
        )
