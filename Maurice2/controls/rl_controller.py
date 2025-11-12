# rl_rollrate_train.py
# Train an RL policy to minimize |w3| by actuating aileron deflection.
# Measurements come from RocketPy; PPO directly commands the aileron via SilSim, while
# the observer update stays AX + BU - L(CX - Y). We do not modify your dynamics files.

import os
import threading
import queue
from pathlib import Path
import numpy as np
import time as _time

# Gym / SB3
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

# Your code (DO NOT MODIFY)
from control_algorithm import Controls
from silsim import SilSim
from fin import Fins

DEG = np.pi / 180.0


class RocketPyRollEnv(gym.Env):
    """
    A Gym environment that runs a RocketPy SIL simulation and synchronizes with PPO step-by-step.

    Synchronization strategy (no changes to your dynamics files):
      - We create a SilSim instance and monkey-patch its controller_function to a wrapper that:
          * converts RocketPy state -> xhat using your mapping
          * builds measurement y from RocketPy state
          * computes A, B, C, and observer update: xdot = A x + B u + aT + aG - L(Cx - y)
          * normalizes quaternion
          * PUTS (obs, reward, info, done) into an obs_queue
          * WAITS for an action from the act_queue
          * sets fins.aileronAngles = action and continues
      - The Gym env.step(action) pushes 'action' into the act_queue, then waits to receive
        the next (obs, reward, done) from the obs_queue.

    Observation: xhat = [w1, w2, w3, v1, v2, v3, qw, qx, qy, qz]
    Action:      aileron deflection delta1 in radians in [-8°, +8°]
    Reward:      -( |w3| ) - λ_u u^2 - λ_du (Δu)^2
    Episode:     ends at RocketPy flight termination (apogee/landing) as handled by SilSim/Flight.
    """

    def __init__(self, dt=1/40.0, lam_u=1e-3, lam_du=5e-3, seed=42):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.dt = float(dt)
        self.lam_u = lam_u
        self.lam_du = lam_du

        # Action bounds (radians)
        self.u_min = -8.0 * DEG
        self.u_max = +8.0 * DEG
        self.action_space = spaces.Box(low=np.array([self.u_min], dtype=np.float32),
                                       high=np.array([self.u_max], dtype=np.float32),
                                       dtype=np.float32)

        # Observation (10,)
        high = np.array([np.inf]*10, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Queues to synchronize the RocketPy controller thread with the RL loop
        self._act_q = queue.Queue(maxsize=1)   # RL -> controller (action)
        self._obs_q = queue.Queue(maxsize=1)   # controller -> RL (obs, reward, done, info)

        self._thread = None
        self._sim = None
        self._controller = None
        self._prev_u = 0.0
        self._started = False

    def _controller_function_rl(self, time, sampling_rate, state, state_history, observed_variables, interactive_objects):
        """Runs each controller tick inside RocketPy; bridges with PPO via queues."""
        try:
            fins: Fins = interactive_objects[0]  # same assumption as your SilSim
            # xhat from RocketPy state
            xhat = self._sim.rocketpy_state_to_xhat(state).astype(float)

            # measurement y (gyro + quat) built from RocketPy state
            y = self._sim.make_measurement_from_rocketpy(state=state, time=time)

            # Previous input from fins; ensure scalar
            u_prev = float(getattr(fins, "aileronAngles", 0.0))

            # Compute system matrices at previous input
            self._controller.computeAB(t=time, xhat=xhat, u=u_prev)
            self._controller.computeC(xhat=xhat, u=u_prev)
            A = np.array(self._controller.A, dtype=float)
            B = np.array(self._controller.B, dtype=float)
            C = np.array(self._controller.C, dtype=float)
            L = np.array(self._controller.L, dtype=float)

            aT = self._controller.get_thrust_accel(t=time)
            aG = self._controller.get_gravity_accel(xhat=xhat)

            # Wait for RL action (with safe fallback to u_prev if queue is empty)
            try:
                u = float(self._act_q.get(timeout=30.0))  # more generous on first handshake
            except queue.Empty:
                u = u_prev  # HOLD previous command if RL didn't send in time

            # Clamp and apply to fins
            u = float(np.clip(u, self.u_min, self.u_max))
            fins.aileronAngles = u
            u_vec = np.array([u], dtype=float)

            # Observer/Luenberger update using RocketPy measurement y
            xdot = A @ xhat + B @ u_vec + aT + aG - L @ (C @ xhat - y)
            xhat_next = xhat + self._controller.dt * xdot

            # Normalize quaternion
            qn = np.linalg.norm(xhat_next[6:10])
            if qn > 1e-12:
                xhat_next[6:10] = xhat_next[6:10] / qn
            else:
                xhat_next[6:10] = np.array([1.0, 0.0, 0.0, 0.0])

            # Reward
            w3 = float(xhat_next[2])
            du = u - self._prev_u
            reward = -(abs(w3) + self.lam_u * (u**2) + self.lam_du * (du**2))
            self._prev_u = u

            # Done logic (same as before)
            v3_body = float(xhat_next[5])
            past_burnout = (time > self._controller.t_motor_burnout)
            done = (past_burnout and v3_body <= 0.0) or (time >= self._controller.t_estimated_apogee)

            info = {"t": time, "w3": w3, "u": u, "v3": v3_body}

            # Send next obs to RL
            try:
                self._obs_q.put((xhat_next.astype(np.float32), reward, done, info), timeout=10.0)
            except queue.Full:
                pass

            return None

        except Exception as e:
            # Always signal the env so it doesn't hang
            err_info = {"error": f"controller_fn exception: {type(e).__name__}: {e}"}
            try:
                self._obs_q.put((np.zeros(10, dtype=np.float32), -1.0, True, err_info), timeout=2.0)
            except queue.Full:
                pass
            # Also re-raise so RocketPy logs it
            raise


    # ---------- Thread target that runs RocketPy flight ----------
    def _run_flight_thread(self):
        """
        Launch a RocketPy Flight via SilSim.run(), but first replace SilSim.controller_function
        with our RL-bridged function so each tick syncs with PPO.
        """
        # Build your Controls and L gains exactly as you usually do
        self._controller = Controls(dt=self.dt,
                                    x0=np.array([0,0,0, 0,0,0, 1,0,0,0], dtype=float),
                                    u0=np.array([0.0], dtype=float),
                                    t_launch_rail_clearance=0.308)
        self._controller.setup_EOM()

        # K schedule is irrelevant to PPO (PPO decides), but kept to preserve context
        self._controller.set_K_params(K_pre_max=0.5, K_pre_min=0.5,
                                      K_post_max=0.5, K_post_min=0.5,
                                      pre_width=3.0, post_width=8.0,
                                      pre_v3_mid=100.0, post_v3_mid=90.0)

        # Observer L (same defaults as before)
        lw = 5e-3
        lq = 5e-3
        self._controller.buildL(lw=lw, lqw=lq, lqx=lq, lqy=lq, lqz=lq)

        # Create SilSim and monkey-patch controller_function to our RL-bridged version
        self._sim = SilSim(sampling_rate=1.0/self.dt, controller=self._controller)
        # Bind our wrapper as the instance method used by RocketPy
        # RocketPy will call this at 'sampling_rate' Hz
        self._sim.controller_function = self._controller_function_rl.__get__(self, RocketPyRollEnv)

        # Run the flight (blocking inside this thread); will keep calling controller_function.
        try:
            self._sim.run(sampling_rate=1.0/self.dt)
        except Exception as e:
            # If something goes wrong, signal termination to the RL loop
            try:
                self._obs_q.put((np.zeros(10, dtype=np.float32), -1.0, True, {"error": str(e)}), timeout=2.0)
            except queue.Full:
                pass

    # ---------- Gym API ----------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # clear queues
        while not self._act_q.empty():
            try: self._act_q.get_nowait()
            except queue.Empty: break
        while not self._obs_q.empty():
            try: self._obs_q.get_nowait()
            except queue.Empty: break

        self._prev_u = 0.0
        self._started = True

        # Start RocketPy in a background thread; we interact via queues
        self._thread = threading.Thread(target=self._run_flight_thread, daemon=True)
        self._thread.start()

        # For the very first tick, the controller will block waiting on our first action.
        # So we must immediately provide a neutral action (0) to get the first observation out.
        try:
            self._act_q.put(0.0, timeout=5.0)
        except queue.Full:
            pass

        # Wait for first observation from the controller
        obs, reward, done, info = self._obs_q.get(timeout=10.0)
        return obs, {}

    def step(self, action):
        if not self._started:
            raise RuntimeError("Call reset() before step().")

        u = float(np.clip(action[0], self.u_min, self.u_max))
        # Send action to controller for the next tick
        self._act_q.put(u, timeout=10.0)

        # Receive next observation, reward, and done from controller
        obs, reward, done, info = self._obs_q.get(timeout=20.0)
        terminated = bool(done)
        truncated = False
        return obs, float(reward), terminated, truncated, info

    def render(self):
        # Not needed; prints can be added from info if desired
        pass

    def close(self):
        # Best-effort join
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)


def main():
    # --- Build env ---
    env = RocketPyRollEnv(dt=1/40.0)
    env = Monitor(env)

    # --- PPO config ---
    policy_kwargs = dict(net_arch=[128, 128])
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./tb_rollrate_rp/",
    )

    # Evaluate/early-stop on good average reward (≈ small |w3|)
    callback = EvalCallback(
        env,
        best_model_save_path="./rl_rollrate_rp_best/",
        log_path="./rl_rollrate_rp_logs/",
        eval_freq=5000,
        deterministic=True,
        render=False,
        callback_after_eval=StopTrainingOnRewardThreshold(reward_threshold=-0.02, verbose=1),
        n_eval_episodes=3,
    )

    timesteps = 150_000
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)
    model.save("ppo_rollrate_aileron_rp.zip")
    print("Training complete. Saved to ppo_rollrate_aileron_rp.zip")

    # Quick eval rollout
    obs, _ = env.reset(seed=123)
    ret = 0.0
    for _ in range(1500):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        ret += r
        if term or trunc:
            break
    print(f"Eval return: {ret:.3f}; final |w3|={abs(info.get('w3',0.0)):.4f} rad/s, u={info.get('u',0.0)/DEG:.2f} deg")


if __name__ == "__main__":
    main()
