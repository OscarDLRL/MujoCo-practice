#!/usr/bin/env python3
"""Run PMP / LQG / MPC controllers on a quadruped robot in MuJoCo with rendering.

Examples:
    python examples/run_mujoco.py
    python examples/run_mujoco.py --controller pmp
    python examples/run_mujoco.py --controller mpc --robot-name mini_cheetah
    python examples/run_mujoco.py --controller lqg --robot-name go2 --teleop
    python examples/run_mujoco.py --controller all --robot-name mini_cheetah --no-render
    python examples/run_mujoco.py --controller lqg --disturbance persistent --teleop

Teleop keys:
    Arrow Up / Arrow Down    -> forward / backward
    Arrow Left / Arrow Right -> yaw left / right
    z / c                    -> lateral left / right
    Space                    -> zero commands
"""

import sys
import os
import argparse
import threading
import select
import numpy as np
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gym_quadruped.quadruped_env import QuadrupedEnv

from src.dynamics import QuadrupedDynamics
from src.estimator_ekf import OrientationEKF
from src.controller_pmp import PontryaginController
from src.controller_lqg import LQGController
from src.controller_mpc import MPCController


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
])


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
def get_state(env) -> np.ndarray:
    """Extract x = [p(3), v(3), rpy(3), ω_body(3)] from the MuJoCo env."""
    p = env.base_pos.copy()
    v = env.base_lin_vel(frame="world")
    rpy = env.base_ori_euler_xyz.copy()
    omega = env.base_ang_vel(frame="base")
    return np.concatenate([p, v, rpy, omega])


def grf_to_torques(env, grfs: np.ndarray, contact: np.ndarray) -> np.ndarray:
    """Convert ground reaction forces to joint torques via Jacobian transpose."""
    tau = np.zeros(env.mjModel.nu)

    try:
        jacobians = env.feet_jacobians(frame="world")
    except Exception:
        return tau

    for i, leg in enumerate(["FL", "FR", "RL", "RR"]):
        if not contact[i]:
            continue

        f_leg = grfs[3 * i: 3 * i + 3]
        J_full = jacobians[leg]
        leg_idx = env.legs_qvel_idx[leg]
        J_leg = J_full[:, leg_idx]
        tau_leg = -J_leg.T @ f_leg

        tau_idx = env.legs_tau_idx[leg]
        tau[tau_idx] = tau_leg

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
    """
    x_ref = dyn.standing_state(height=height)
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
def save_single_run_plot(result, controller_name, robot_name, disturbance_type, x_ref_nominal):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("results", exist_ok=True)

    log_t = result["time"]
    log_x = result["state"]
    log_u = result["control"]
    log_dist = result["disturbance"]

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"{controller_name.upper()} — {robot_name} — {disturbance_type}",
        fontsize=14,
        fontweight="bold",
    )

    labels_p = [r"$p_x$", r"$p_y$", r"$p_z$"]
    for i in range(3):
        axes[0].plot(log_t, log_x[:, i], label=labels_p[i])
        axes[0].axhline(x_ref_nominal[i], ls="--", color="gray", lw=0.6)
    axes[0].set_ylabel("Position [m]")
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].grid(True, alpha=0.3)

    labels_v = [r"$v_x$", r"$v_y$", r"$v_z$"]
    for i in range(3):
        axes[1].plot(log_t, log_x[:, 3 + i], label=labels_v[i])
    axes[1].set_ylabel("Velocity [m/s]")
    axes[1].legend(ncol=3, fontsize=8)
    axes[1].grid(True, alpha=0.3)

    labels_o = ["roll", "pitch", "yaw"]
    for i in range(3):
        axes[2].plot(log_t, np.degrees(log_x[:, 6 + i]), label=labels_o[i])
    axes[2].set_ylabel("Orientation [deg]")
    axes[2].legend(ncol=3, fontsize=8)
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(log_t, np.linalg.norm(log_u, axis=1), label="||GRFs||")
    axes[3].fill_between(log_t, 0, log_dist * 2, alpha=0.25, label="disturbance")
    axes[3].set_ylabel("Force [N]")
    axes[3].set_xlabel("Time [s]")
    axes[3].legend(fontsize=8)
    axes[3].grid(True, alpha=0.3)

    for ax in axes:
        if disturbance_type == "impulse":
            ax.axvspan(2.0, 2.15, alpha=0.1)
        elif disturbance_type == "persistent":
            ax.axvspan(2.0, log_t[-1], alpha=0.05)

    plt.tight_layout()
    path = f"results/mujoco_{controller_name}_{robot_name}_{disturbance_type}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Plot saved: {path}")


def save_comparison_plot(results, robot_name, disturbance_type):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("results", exist_ok=True)

    colors = {"pmp": "#e74c3c", "lqg": "#2ecc71", "mpc": "#3498db"}

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(
        f"Controller Comparison — {robot_name} — {disturbance_type}",
        fontsize=14,
        fontweight="bold",
    )

    for name, data in results.items():
        t = data["time"]
        x = data["state"]
        u = data["control"]

        pos_err = np.linalg.norm(x[:, :3] - np.array([0.0, 0.0, ROBOT_HIP_HEIGHT]), axis=1)
        vel_err = np.linalg.norm(x[:, 3:6], axis=1)
        u_norm = np.linalg.norm(u, axis=1)

        axes[0].plot(t, pos_err, color=colors[name], label=name.upper(), lw=1.5)
        axes[1].plot(t, vel_err, color=colors[name], lw=1.5)
        axes[2].plot(t, u_norm, color=colors[name], lw=1.2)

    axes[0].set_ylabel("Position error [m]")
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
    path = f"results/mujoco_comparison_{robot_name}_{disturbance_type}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Comparison plot saved: {path}")


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
):
    print(f"\n{'=' * 60}")
    print(f"  Controller:   {controller_name.upper()}")
    print(f"  Robot:        {robot_name}")
    print(f"  Teleop:       {teleop_enabled}")
    print(f"  Duration:     {duration}s")
    print(f"  Disturbance:  {disturbance_type}")
    print(f"{'=' * 60}\n")

    state_obs_names = tuple(QuadrupedEnv.ALL_OBS)

    env = QuadrupedEnv(
        robot=robot_name,
        scene="flat",
        sim_dt=0.002,
        base_vel_command_type="human",
        state_obs_names=state_obs_names,
    )

    _ = env.reset(random=False)
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

    x_ref = build_reference_state(dyn, height=ROBOT_HIP_HEIGHT, vx=0.0, vy=0.0, wz=0.0)
    u_ref = dyn.standing_control()
    controller = build_controller(controller_name, dyn, Q, R, Q_f, x_ref)

    ori_ekf = OrientationEKF(dt=env.mjModel.opt.timestep)

    sim_dt = env.mjModel.opt.timestep
    ctrl_dt = 0.01
    ctrl_steps = max(1, int(ctrl_dt / sim_dt))
    n_steps = int(duration / sim_dt)

    log_t, log_x, log_u, log_err, log_dist = [], [], [], [], []
    current_grfs = u_ref.copy()

    print(f"  Sim dt: {sim_dt}s, Ctrl rate: {1 / ctrl_dt:.0f} Hz, Total steps: {n_steps}")
    print("  Starting simulation...\n")

    try:
        for step in range(n_steps):
            t = step * sim_dt

            x = get_state(env)
            contact = get_contacts(env)
            r_feet = get_feet_world(env)

            cmd_vx = teleop.vx if teleop_enabled else 0.0
            cmd_vy = teleop.vy if teleop_enabled else 0.0
            cmd_wz = teleop.wz if teleop_enabled else 0.0

            x_ref = build_reference_state(
                dyn,
                height=ROBOT_HIP_HEIGHT,
                vx=cmd_vx,
                vy=cmd_vy,
                wz=cmd_wz,
            )

            try:
                if hasattr(env, "target_base_vel"):
                    env.target_base_vel[:] = np.array([cmd_vx, cmd_vy, 0.0])
                if hasattr(env, "target_base_ang_vel"):
                    env.target_base_ang_vel[:] = np.array([0.0, 0.0, cmd_wz])
                if hasattr(env, "ref_base_lin_vel"):
                    env.ref_base_lin_vel = cmd_vx
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
                    _A_c_new, _B_c_new = dyn.continuous_AB(x, contact, r_feet)
                    _A_d_new, _B_d_new = dyn.discretize(_A_c_new, _B_c_new)
                    _g_d = dyn.gravity_vector()
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

            tau = grf_to_torques(env, current_grfs, contact)
            _, _, terminated, _, _ = env.step(action=tau)

            if render:
                env.render()

            log_t.append(t)
            log_x.append(x.copy())
            log_u.append(current_grfs.copy())
            log_err.append(np.linalg.norm(x[:6] - x_ref[:6]))
            log_dist.append(np.linalg.norm(dist))

            if step % int(1.0 / sim_dt) == 0:
                pos_err = np.linalg.norm(x[:3] - x_ref[:3])
                vel_err = np.linalg.norm(x[3:6] - x_ref[3:6])
                print(
                    f"  t={t:5.1f}s | pos_err={pos_err:.4f}m | "
                    f"vel_err={vel_err:.4f}m/s | "
                    f"height={x[2]:.3f}m | "
                    f"vx={x[3]:+.3f} | vy={x[4]:+.3f} | wz={x[11]:+.3f} | "
                    f"cmd=({cmd_vx:+.2f},{cmd_vy:+.2f},{cmd_wz:+.2f})"
                )

            if terminated:
                print(f"  Terminated at t={t:.2f}s")
                _ = env.reset(random=False)
                if render:
                    env.render()

    except KeyboardInterrupt:
        print("\n  Interrupted by user.")

    finally:
        teleop.quit_requested = True
        env.close()

    log_t = np.array(log_t) if len(log_t) > 0 else np.zeros(1)
    log_x = np.array(log_x) if len(log_x) > 0 else np.zeros((1, 12))
    log_u = np.array(log_u) if len(log_u) > 0 else np.zeros((1, 12))
    log_err = np.array(log_err) if len(log_err) > 0 else np.zeros(1)
    log_dist = np.array(log_dist) if len(log_dist) > 0 else np.zeros(1)

    result = {
        "time": log_t,
        "state": log_x,
        "control": log_u,
        "error": log_err,
        "disturbance": log_dist,
    }

    if save_log and len(log_t) > 1:
        x_ref_nominal = build_reference_state(dyn, height=ROBOT_HIP_HEIGHT, vx=0.0, vy=0.0, wz=0.0)
        save_single_run_plot(result, controller_name, robot_name, disturbance_type, x_ref_nominal)

    print(f"\n  --- {controller_name.upper()} Summary ---")
    print(f"  Position/velocity RMSE: {np.sqrt(np.mean(log_err**2)):.4f}")
    print(f"  Max error: {np.max(log_err):.4f}")
    print(f"  Mean GRF norm: {np.mean(np.linalg.norm(log_u, axis=1)):.1f} N")

    return result


# =====================================================================
# Comparison mode
# =====================================================================
def run_comparison(
    render: bool,
    duration: float,
    disturbance_type: str,
    robot_name: str,
):
    results = {}
    for name in ["pmp", "lqg", "mpc"]:
        results[name] = run(
            name,
            robot_name=robot_name,
            teleop_enabled=False,
            render=render,
            duration=duration,
            disturbance_type=disturbance_type,
            save_log=False,
        )

    save_comparison_plot(results, robot_name, disturbance_type)

    print(f"\n{'=' * 60}")
    print(f"  COMPARISON SUMMARY ({disturbance_type})")
    print(f"{'=' * 60}")
    print(f"  {'Controller':<12} {'RMSE':>10} {'Mean ||u||':>12}")
    print(f"  {'-' * 38}")
    for name, data in results.items():
        print(
            f"  {name.upper():<12} "
            f"{np.sqrt(np.mean(data['error']**2)):>10.4f} "
            f"{np.mean(np.linalg.norm(data['control'], axis=1)):>12.1f}"
        )
    print(f"{'=' * 60}")


# =====================================================================
# CLI
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quadruped control with MuJoCo rendering")
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
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument(
        "--disturbance",
        default="impulse",
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
        )