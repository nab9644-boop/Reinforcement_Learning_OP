"""
hw3_sqp_cartpole.py  —  ROB 6323  HW3 Exercise 2
=============================================
SQP-based model predictive control (MPC) for the cart-pole swing-up,
with Pinocchio (dynamics), qpsolvers/ProxQP (QPs) and MuJoCo (simulation).

CONVENTIONS
-----------
State:   x = (p, θ, ṗ, θ̇)   x[:2] = qpos, x[2:] = qvel   (MuJoCo names)
         p = cart position [m], θ = pole angle [rad]
         θ = 0 upright (goal),  θ = π hanging (start)
Control: u = horizontal force on the cart [N]
n = stage (0 … N),  k = SQP iteration.

Decision vector, stage by stage:
   z = (x_0, u_0, x_1, u_1, ..., x_{N-1}, u_{N-1}, x_N),   n_z = 5N + 4
   z[idx_x(n)] = x_n,   z[idx_u(n)] = u_n

Dynamics: the OCP model f is Pinocchio + explicit Euler (discrete_dynamics,
given).  MuJoCo, the "robot", integrates with its own semi-implicit Euler, so
the two do not match exactly; MPC closes the loop on the difference.  Both
use the same Δt.

USAGE
-----
  python hw3_sqp_cartpole.py                    # MPC with PARAMS
  python hw3_sqp_cartpole.py --display          # replay in the MuJoCo viewer (use mjpython instead of python in macOS)
"""

import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pinocchio as pin
import mujoco
import qpsolvers

N_X = 4
N_U = 1
S_ACT = np.array([1.0, 0.0])   # the force acts on the slider only


# ── GIVEN: model loading ──────────────────────────────────────────────────────

MJCF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cartpole.xml')

def load_models(params):
    """Load cartpole.xml in Pinocchio and MuJoCo.

    Returns (pin_model, pin_data, mj_model).
    MuJoCo's timestep and force limit are set from params['dt'] and
    params['u_max'] (no limit if u_max is infinite), so the simulator and the
    OCP always share them.
    """
    pin_model = pin.buildModelsFromMJCF(MJCF_PATH)[0]
    pin_data = pin_model.createData()
    mj_model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    mj_model.opt.timestep = params['dt']
    if np.isfinite(params['u_max']):
        mj_model.actuator_ctrlrange[0] = [-params['u_max'], params['u_max']]
    else:
        mj_model.actuator_ctrllimited[0] = 0
    return pin_model, pin_data, mj_model


# ── GIVEN: parameters ─────────────────────────────────────────────────────────

PARAMS = dict(
    dt=0.02,                    # OCP and MuJoCo timestep [s]
    N=50,                       # OCP horizon: 50 × 0.02 s = 1 s
    u_max=20.0,               # force bound [N]; start with no bound
    x_init=np.array([0., np.pi, 0., 0.]),
    x_goal=np.zeros(4),
    # MPC
    T_sim=8.0,                  # closed-loop simulation length [s]
    mpc_max_iter=10,            # SQP iterations per control step
    mpc_max_iter_first=100,     # first step is a cold start: allow more
    mpc_tol_c=1e-6,             # SQP stopping tolerances inside MPC
    mpc_tol_kkt=1e-4,
    Q=np.diag([1.0, 10.0, 0.1, 0.1]),      # (p, theta, p_dot, theta_dot)
    R=np.array([[0.01]]),
    QN=np.diag([100.0, 100.0, 10.0, 10.0]),
)
    
def cost(z, params):
    """J(z) and its gradient grad J(z)."""
    N, xg = params['N'], params['x_goal']
    Q, R, QN = params['Q'], params['R'], params['QN']
    J = 0.0
    g = np.zeros_like(z)
    for n in range(N):
        dx = z[idx_x(n)] - xg
        u = z[idx_u(n)]
        J += 0.5 * dx @ Q @ dx + 0.5 * u @ R @ u
        g[idx_x(n)] = Q @ dx
        g[idx_u(n)] = R @ u
    dxN = z[idx_x(N)] - xg
    J += 0.5 * dxN @ QN @ dxN
    g[idx_x(N)] = QN @ dxN
    return J, g


def cost_hessian(params):
    """Constant, block-diagonal Hessian of J (quadratic cost)."""
    N = params['N']
    H = np.zeros((n_z_total(N), n_z_total(N)))
    for n in range(N):
        H[idx_x(n), idx_x(n)] = params['Q']
        H[idx_u(n), idx_u(n)] = params['R']
    H[idx_x(N), idx_x(N)] = params['QN']
    return H
# ── GIVEN: indexing ───────────────────────────────────────────────────────────

def idx_x(n):
    """Slice of z holding x_n.  Each stage block is [x_n | u_n] = 5 entries."""
    return slice(n * (N_X + N_U), n * (N_X + N_U) + N_X)


def idx_u(n):
    """Slice of z holding u_n."""
    return slice(n * (N_X + N_U) + N_X, n * (N_X + N_U) + N_X + N_U)


def n_z_total(N):
    """Size of the decision vector z."""
    return N * (N_X + N_U) + N_X


def rest_guess(params):
    """Cold-start guess: x_n = x_init for every stage, u_n = 0."""
    N = params['N']
    z0 = np.zeros(n_z_total(N))
    for n in range(N + 1):
        z0[idx_x(n)] = params['x_init']
    return z0


# ── GIVEN (example of the Pinocchio API): discrete dynamics ──────────────────

def discrete_dynamics(x, u, pin_model, pin_data, dt):
    """x_{n+1} = f(x_n, u_n): Pinocchio forward dynamics + explicit Euler.

        a     = pin.aba(model, data, qpos, qvel, τ),   τ = S_ACT u
        qpos+ = qpos + dt * qvel
        qvel+ = qvel + dt * a
    """
    qpos, qvel = x[:2], x[2:]
    tau = S_ACT * float(np.squeeze(u))
    a = pin.aba(pin_model, pin_data, qpos, qvel, tau)
    return np.concatenate([qpos + dt * qvel, qvel + dt * a])


def dynamics_derivatives(x, u, pin_model, pin_data, dt):
    """Jacobians A_n = ∂f/∂x (4×4) and B_n = ∂f/∂u (4×1) of discrete_dynamics.

    pin.computeABADerivatives fills data.ddq_dq = ∂a/∂qpos,
    data.ddq_dv = ∂a/∂qvel and data.Minv = ∂a/∂τ = M⁻¹.  Then
        A_n = [[ I,          dt I         ],     B_n = [[ 0               ],
               [ dt ∂a/∂q,   I + dt ∂a/∂v ]]            [ dt M⁻¹ S_ACT    ]]
    """
    qpos, qvel = x[:2], x[2:]
    tau = S_ACT * float(np.squeeze(u))
    pin.computeABADerivatives(pin_model, pin_data, qpos, qvel, tau)
    I2 = np.eye(2)
    A_n = np.block([[I2,                     dt * I2],
                    [dt * pin_data.ddq_dq,   I2 + dt * pin_data.ddq_dv]])
    B_n = np.zeros((4, 1))
    B_n[2:, 0] = dt * (pin_data.Minv @ S_ACT)
    return A_n, B_n

# ── OCP pieces: constraints, bounds ──────────────────────────────────────────

def constraints(z, x_init, pin_model, pin_data, params, jac=True):
    """Equality constraints c(z) (initial state + dynamics) and, if jac, their Jacobian Jc."""
    N, dt = params['N'], params['dt']
    n_c = N_X * (N + 1)
    c = np.zeros(n_c)
    Jc = np.zeros((n_c, n_z_total(N))) if jac else None
    rows = lambda k: slice(k * N_X, (k + 1) * N_X)

    c[rows(0)] = z[idx_x(0)] - x_init
    if jac:
        Jc[rows(0), idx_x(0)] = np.eye(N_X)
    for n in range(N):
        x_n, u_n = z[idx_x(n)], z[idx_u(n)]
        c[rows(n + 1)] = z[idx_x(n + 1)] - discrete_dynamics(x_n, u_n, pin_model, pin_data, dt)
        if jac:
            A_n, B_n = dynamics_derivatives(x_n, u_n, pin_model, pin_data, dt)
            Jc[rows(n + 1), idx_x(n + 1)] = np.eye(N_X)
            Jc[rows(n + 1), idx_x(n)] = -A_n
            Jc[rows(n + 1), idx_u(n)] = -B_n
    return (c, Jc) if jac else c


def bounds(params):
    """Box constraints lb <= z <= ub: only the controls are bounded."""
    N, u_max = params['N'], params['u_max']
    lb = np.full(n_z_total(N), -np.inf)
    ub = np.full(n_z_total(N), np.inf)
    for n in range(N):
        lb[idx_u(n)] = -u_max
        ub[idx_u(n)] = u_max
    return lb, ub


# ── SQP: QP step, merit function, line search ────────────────────────────────

BIG = 1e20   # stands in for "infinity" in the QP bounds


def bound_violation(z, lb, ub):
    return np.sum(np.maximum(z - ub, 0)) + np.sum(np.maximum(lb - z, 0))


def merit(z, sigma, x_init, pin_model, pin_data, params, lb, ub):
    """l1 merit function: phi = J + sigma * (||c||_1 + bound violation)."""
    J, _ = cost(z, params)
    c = constraints(z, x_init, pin_model, pin_data, params, jac=False)
    return J + sigma * (np.sum(np.abs(c)) + bound_violation(z, lb, ub))


def solve_qp_step(z, H, g, c, Jc, lb, ub):
    """QP:  min 1/2 p'Hp + g'p   s.t.  Jc p = -c,   u-bounds as G p <= h (only the u entries)."""
    iu = np.where(np.isfinite(ub))[0]                 # indices of the bounded entries (the u's)
    if len(iu) == 0:                                  # no bounds (u_max = inf)
        prob = qpsolvers.Problem(P=H, q=g, A=Jc, b=-c)
    else:
        G = np.zeros((2 * len(iu), len(z)))
        G[np.arange(len(iu)), iu] = 1.0               #  p_u <= ub - z_u
        G[len(iu) + np.arange(len(iu)), iu] = -1.0    # -p_u <= z_u - lb
        h = np.concatenate([ub[iu] - z[iu], z[iu] - lb[iu]])
        prob = qpsolvers.Problem(P=H, q=g, A=Jc, b=-c, G=G, h=h)
    sol = qpsolvers.solve_problem(prob, solver="proxqp")
    if not sol.found:                                  # retry with looser settings
        sol = qpsolvers.solve_problem(prob, solver="proxqp", eps_abs=1e-5, max_iter=20000)
    if not sol.found:                                  # give up on this SQP iteration
        return None, None, None 
    mu = G.T @ sol.z if len(iu) else np.zeros_like(z)  # bound multipliers, mapped back onto z
    return sol.x, sol.y, mu


def solve_ocp(z0, x_init, pin_model, pin_data, params, max_iter, c1=1e-4, rho=0.5):
    """SQP with l1-merit backtracking line search. Returns z and an info dict."""
    z = z0.copy()
    H = cost_hessian(params)
    lb, ub = bounds(params)
    sigma = 1.0
    info = dict(iters=0, converged=False, alphas=[])
    for k in range(max_iter):
        J, g = cost(z, params)
        c, Jc = constraints(z, x_init, pin_model, pin_data, params)
        p, lam, mu = solve_qp_step(z, H, g, c, Jc, lb, ub)
        if p is None:                                   # QP failed: keep the current plan
            info['qp_failed'] = True
            break

        # stopping test: constraints satisfied and KKT residual small
        kkt = np.max(np.abs(g + Jc.T @ lam + mu))
        if np.max(np.abs(c)) < params['mpc_tol_c'] and kkt < params['mpc_tol_kkt']:
            info.update(iters=k, converged=True)
            break

        # merit weight must exceed the largest multiplier
        sigma = max(sigma, 1.1 * max(np.max(np.abs(lam)), np.max(np.abs(mu))))
        viol = np.sum(np.abs(c)) + bound_violation(z, lb, ub)
        phi0 = J + sigma * viol
        D = g @ p - sigma * viol                 # directional derivative of phi along p

        alpha = 1.0
        while (merit(z + alpha * p, sigma, x_init, pin_model, pin_data, params, lb, ub)
               > phi0 + c1 * alpha * D) and alpha > 1e-8:
            alpha *= rho
        z = z + alpha * p
        info['alphas'].append(alpha)
        info['iters'] = k + 1
    return z, info


def shift_warm_start(z_prev, params):
    """Shift the previous plan forward by one stage; repeat the last stage at the end."""
    N = params['N']
    z = z_prev.copy()
    for n in range(N - 1):
        z[idx_x(n)] = z_prev[idx_x(n + 1)]
        z[idx_u(n)] = z_prev[idx_u(n + 1)]
    z[idx_x(N - 1)] = z_prev[idx_x(N)]
    z[idx_u(N - 1)] = z_prev[idx_u(N - 1)]
    z[idx_x(N)] = z_prev[idx_x(N)]
    return z


# ── TODO: MPC ───────────────────────────────────────────────────────────────────

def mpc_step(x_meas, z_prev, pin_model, pin_data, params):
    p = dict(params, x_init=x_meas)
    if z_prev is None:                                   # first step: cold start
        z0 = rest_guess(p)
        max_iter = params['mpc_max_iter_first']
    else:                                                # later steps: warm start
        z0 = shift_warm_start(z_prev, params)
        max_iter = params['mpc_max_iter']
    z0[idx_x(0)] = x_meas
    z, info = solve_ocp(z0, x_meas, pin_model, pin_data, p, max_iter)
    #print(f"  t-step: {info['iters']} SQP iters, converged={info['converged']}, |theta|={abs(x_meas[1]):.2f}")
    return float(z[idx_u(0)][0]), z


# ── GIVEN: closed loop in MuJoCo ─────────────────────────────────────────────

def simulate_mpc(mj_model, pin_model, pin_data, params):
    """Run MPC on MuJoCo for params['T_sim'] seconds.

    Every Δt: measure (qpos, qvel), call mpc_step, apply u_0, let MuJoCo
    integrate one step.  Returns a dict with t (n+1,), X (n+1, 4), U (n,) and
    t_solve (n,), the wall time of each MPC step [s].
    """
    dt = params['dt']
    n_steps = int(round(params['T_sim'] / dt))
    d = mujoco.MjData(mj_model)
    d.qpos[:] = params['x_init'][:2]
    d.qvel[:] = params['x_init'][2:]

    X = [np.concatenate([d.qpos, d.qvel])]
    U, t_solve = [], []
    z = None
    for _ in range(n_steps):
        t0 = time.perf_counter()
        u, z = mpc_step(X[-1], z, pin_model, pin_data, params)
        t_solve.append(time.perf_counter() - t0)
        d.ctrl[:] = [u]
        mujoco.mj_step(mj_model, d)
        U.append(u)
        X.append(np.concatenate([d.qpos, d.qvel]))
    return dict(t=np.arange(n_steps + 1) * dt, X=np.array(X), U=np.array(U),
                t_solve=np.array(t_solve))


def view_trajectory(mj_model, X, dt):
    """Replay a recorded trajectory in the MuJoCo viewer, in real time, until
    the window is closed.  On macOS run the script with `mjpython`."""
    import mujoco.viewer
    d = mujoco.MjData(mj_model)
    try:
        viewer = mujoco.viewer.launch_passive(mj_model, d)
    except RuntimeError as e:
        print(f"  could not open the viewer ({e}); on macOS use `mjpython`.")
        return
    with viewer as v:
        v.cam.lookat[:] = [0.0, 0.0, 0.0]
        v.cam.distance = 4.0
        v.cam.azimuth = 90.0
        v.cam.elevation = -10.0
        while v.is_running():
            for x in X:
                if not v.is_running():
                    return
                t0 = time.perf_counter()
                d.qpos[:] = x[:2]
                d.qvel[:] = x[2:]
                mujoco.mj_forward(mj_model, d)
                v.sync()
                time.sleep(max(0.0, dt - (time.perf_counter() - t0)))
            time.sleep(1.0)

def run_case(params, label):
    """One full MPC simulation; returns the results plus swing-up time."""
    pin_model, pin_data, mj_model = load_models(params)
    r = simulate_mpc(mj_model, pin_model, pin_data, params)
    th = np.arctan2(np.sin(r['X'][:, 1]), np.cos(r['X'][:, 1]))
    up = np.abs(th) < 0.1
    r['swing'] = next((r['t'][i] for i in range(len(r['t'])) if up[i:].all()), None)
    r['th'], r['label'] = th, label
    print(f"{label}: swing-up {r['swing']} s | mean solve {r['t_solve'].mean()*1e3:.0f} ms"
          f" | max |u| {np.max(np.abs(r['U'])):.1f} N")
    return r


def plot_cases(cases, fname, title):
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(7, 8))
    for r in cases:
        ax[0].plot(r['t'], r['th'], label=r['label'])
        ax[1].step(r['t'][:-1], r['U'], where="post")
        ax[2].plot(r['t'][:-1], r['t_solve'] * 1e3)
    ax[0].set_ylabel("theta [rad]"); ax[0].legend()
    ax[1].set_ylabel("u [N]"); ax[2].set_ylabel("solve time [ms]")
    ax[2].set_xlabel("time [s]"); ax[0].set_title(title)
    for a in ax: a.grid()
    plt.tight_layout(); plt.savefig(fname, dpi=150)
    print("saved", fname)
def main():
    parser = argparse.ArgumentParser(description="SQP-based MPC cart-pole swing-up")
    parser.add_argument('--display', action='store_true',
                        help="replay the run in the MuJoCo viewer (mjpython on macOS)")
    parser.add_argument('--sweep_umax', action='store_true')
    parser.add_argument('--terminal', type=float, default=None)   # value = u_max to test at
    parser.add_argument('--rti', action='store_true')
    args = parser.parse_args()

    params = dict(PARAMS)

    # ── experiments (each one runs alone, then stops) ──
    if args.sweep_umax:
        cases = [run_case(dict(PARAMS, u_max=um, T_sim=12.0), f"u_max = {um}") for um in [9.0, 8.0, 7.0, 6.0]]
        plot_cases(cases, "umax_sweep_fine.png", "Smallest force limit u_max (12 s simulation)"); return
    if args.terminal is not None:
        cases = [run_case(dict(PARAMS, u_max=args.terminal, QN=s * PARAMS['QN']), f"QN x {s}")
                 for s in [0.1, 1.0, 10.0]]
        plot_cases(cases, "terminal_cost.png", f"Terminal cost, u_max = {args.terminal}"); return
    if args.rti:
        cases = [run_case(dict(PARAMS, mpc_max_iter=1), "RTI (1 SQP iter/step)"),
                 run_case(dict(PARAMS, mpc_max_iter=50, mpc_max_iter_first=200), "converged (up to 50 iters)")]
        plot_cases(cases, "rti_vs_converged.png", "Real-time iteration vs fully converged"); return

    # ── default: single run with PARAMS ──
    pin_model, pin_data, mj_model = load_models(params)
    r = simulate_mpc(mj_model, pin_model, pin_data, params)
    t, X, U = r['t'], r['X'], r['U']
    th = np.arctan2(np.sin(X[:, 1]), np.cos(X[:, 1]))          # angle wrapped to (-pi, pi]
    up = np.abs(th) < 0.1
    swing = next((t[i] for i in range(len(t)) if up[i:].all()), None)
    print(f"swing-up time (|theta| < 0.1 rad from then on): {swing} s")
    print(f"mean / max solve time per step: {r['t_solve'].mean()*1e3:.1f} / {r['t_solve'].max()*1e3:.1f} ms")

    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(7, 9))
    ax[0].plot(t, th); ax[0].set_ylabel("theta [rad]")
    ax[1].plot(t, X[:, 0]); ax[1].set_ylabel("cart p [m]")
    ax[2].step(t[:-1], U, where="post"); ax[2].set_ylabel("u [N]")
    ax[3].plot(t[:-1], r['t_solve'] * 1e3); ax[3].set_ylabel("solve time [ms]")
    ax[3].set_xlabel("time [s]")
    for a in ax: a.grid()
    plt.tight_layout(); plt.savefig("mpc_run.png", dpi=150)
    if not args.display:
        plt.show()

    if args.display:
        view_trajectory(mj_model, r['X'], params['dt'])
if __name__ == '__main__':
    main()
