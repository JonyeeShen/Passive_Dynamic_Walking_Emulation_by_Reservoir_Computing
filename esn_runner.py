import time, os, copy
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from collections import deque


def expand_state(state, gamma, time_since_impact, last_impact_n, last_impact_p):
    """
    Expand a 4D walker state with additional contextual information 
    for ESN input.

    Args:
        state (np.ndarray): Current 4D state [theta_ns, theta_s, dot_ns, dot_s].
        gamma (float): Slope parameter.
        time_since_impact (float): Time elapsed since last impact.
        last_impact_n (np.ndarray): Previous state at the last impact (negative step).
        last_impact_p (np.ndarray): Previous state at the last impact (positive step).

    Returns:
        np.ndarray: Extended state vector for ESN input.
    """
    state = np.atleast_2d(state)

    theta_ns = state[:, 0]
    theta_s = state[:, 1]
    theta_dot_ns = state[:, 2]
    theta_dot_s = state[:, 3]

    theta_ns_n, theta_s_n, dot_ns_n, dot_s_n = last_impact_n
    theta_ns_p, theta_s_p, dot_ns_p, dot_s_p = last_impact_p

    extended_in_state = np.column_stack([
        theta_ns,
        theta_s,
        theta_dot_ns,
        theta_dot_s,
        gamma,
        time_since_impact,
        theta_ns_n,
        theta_s_n,
        dot_ns_n,
        dot_s_n,
        theta_ns_p,
        theta_s_p,
        dot_ns_p,
        dot_s_p,
    ])

    return extended_in_state


def impact_detection(state, threshold):
    """
    Detect impact events based on state variables.

    Args:
        state (np.ndarray): Current state [theta_ns, theta_s, dot_ns, dot_s].
        threshold (float): Impact detection threshold.

    Returns:
        bool: 1 if an impact is detected, else 0.
    """
    ns, s, dot_ns, dot_s = state
    if (abs(2 * ns - s) < threshold + 1e-6) and (ns < 0 + 1e-6) and (2 * dot_ns < dot_s + 1e-6):
        return 1
    else:
        return 0


class EchoStateNetwork:
    """
    Minimal Echo State Network (ESN) implementation.

    Attributes:
        W_in (np.ndarray): Input weight matrix.
        W (np.ndarray): Reservoir weight matrix.
        state (np.ndarray): Current reservoir state.
        alpha (float): Leaky integration parameter.
    """

    def __init__(self, input_size, reservoir_size, spectral_radius=0.95, alpha=0.9, density=0.8, input_scale=0.1):
        self.alpha = alpha

        self.W_in = np.random.rand(reservoir_size, input_size) * 2 - 1
        self.W = np.random.rand(reservoir_size, reservoir_size) - 0.5

        self.W_in *= input_scale

        mask = np.random.rand(reservoir_size, reservoir_size) < density
        self.W *= mask

        self.W *= spectral_radius / np.max(np.abs(np.linalg.eigvals(self.W)))
        self.state = (np.random.rand(reservoir_size) * 2 - 1)

    def forward(self, x):
        """
        Forward one step through the ESN.

        Args:
            x (np.ndarray): Input vector.

        Returns:
            np.ndarray: Updated reservoir state.
        """
        u = (self.W_in @ x.T).flatten()
        self.state = self.alpha * self.state + (1 - self.alpha) * np.tanh(self.W @ self.state + u)
        return self.state


def run_one_gamma_with_flag(
    notion, gamma, N,
    static_threshold, static_horizon, static_criteria,
    fluc_threshold, fluc_horizon, fluc_criteria,
    record_washout, impact_threshold,
    sampling_rate, dt,
    test_impact_threshold,
    esn, ridge_model, all_train_data,
    done_flags_dict=None,
    washout=100,
    seed=None,
    print_results=True,
    check_every=64,
):
    """
    Run ESN-based simulation for one gamma value with stopping flags.

    Args:
        notion (str): Identifier for the run.
        gamma (float): Slope parameter.
        N (int): Number of trials.
        static_threshold (int): Threshold for static state detection.
        static_horizon (int): Window size for static detection.
        static_criteria (float): Stability threshold for static detection.
        fluc_threshold (int): Threshold for fluctuation detection.
        fluc_horizon (int): Window size for fluctuation detection.
        fluc_criteria (float): Criteria for fluctuation detection.
        record_washout (int): Number of impacts to discard initially.
        impact_threshold (int): Number of impacts required to terminate.
        sampling_rate (float): Sampling rate of the simulation.
        dt (float): Timestep size.
        test_impact_threshold (float): Threshold for impact detection.
        esn (EchoStateNetwork): Trained ESN instance.
        ridge_model (sklearn.linear_model.Ridge): Linear readout model.
        all_train_data (list): Training trajectories for initialization.
        done_flags_dict (dict): Shared dictionary to indicate early stopping.
        washout (int): Number of washout steps.
        seed (int): Random seed.
        print_results (bool): Whether to print logs.
        check_every (int): Frequency of checking shared flags.

    Returns:
        dict: Simulation results including impact intervals, 
              predictions, and reach status.
    """
    gamma_key = float(f"{gamma:.5f}")

    # Early exit if gamma already completed
    if done_flags_dict is not None and done_flags_dict.get(gamma_key, False):
        return {
            'gamma': float(gamma),
            'impact_intervals': [],
            'test_pred': None,
            'reached': False,
            'skipped': True,
        }

    rng = np.random.default_rng(seed if seed is not None else (abs(hash((gamma, time.time()))) % (2**32-1)))

    # Deepcopy ESN for isolation across processes
    local_esn = copy.deepcopy(esn)
    local_model = ridge_model

    coef_T = local_model.coef_.T  # shape: (reservoir_size, 4)

    test_duration = int(4.3 * impact_threshold * sampling_rate)

    reached = False
    reached_sim = None
    mean_interval = None
    std_interval = None
    impact_intervals = None
    test_pred = None

    esn_input_buf = np.empty((1, 14), dtype=np.float64)

    zero_state = np.zeros(4, dtype=np.float64)
    zero_gamma = float(gamma)
    zero_time = 0.0
    zero_last_n = np.zeros(4, dtype=np.float64)
    zero_last_p = np.zeros(4, dtype=np.float64)

    def fill_esn_input(state4, gam, t_since_imp, last_n, last_p, out):
        """Helper to build ESN input vector."""
        out[0, 0:4] = state4
        out[0, 4]   = gam
        out[0, 5]   = t_since_imp
        out[0, 6:10]  = last_n
        out[0, 10:14] = last_p
        return out

    for sim_id in range(N):
        if done_flags_dict is not None and done_flags_dict.get(gamma_key, False):
            return {
                'gamma': float(gamma),
                'impact_intervals': [],
                'test_pred': None,
                'reached': False,
                'skipped': True,
            }

        # Reset with washout
        fill_esn_input(zero_state, zero_gamma, zero_time, zero_last_n, zero_last_p, esn_input_buf)
        for _ in range(washout):
            local_esn.forward(esn_input_buf)

        # Initialize with random state from training data
        sample_obj = rng.choice(all_train_data)
        ts = sample_obj['time_series']
        rand_i = rng.integers(0, ts.shape[0])
        init_state = ts[rand_i] + rng.normal(0, 0.1, size=ts[rand_i].shape)

        win_std = deque(maxlen=static_horizon)
        win_gap = deque(maxlen=fluc_horizon)

        last_impact_idx = -1
        last_impact_n = np.zeros(4, dtype=np.float64)
        last_impact_p = np.zeros(4, dtype=np.float64)
        impact_count = 0
        impact_intervals_local = []
        last_impact_step = None

        test_prediction_list = [init_state]
        curr_pred = init_state
        win_std.append(curr_pred)
        win_gap.append(curr_pred)

        for i in range(test_duration):
            if (done_flags_dict is not None) and (i % check_every == 0):
                if done_flags_dict.get(gamma_key, False):
                    return {
                        'gamma': float(gamma),
                        'impact_intervals': [],
                        'test_pred': None,
                        'reached': False,
                        'skipped': True,
                    }

            current_state = curr_pred

            if impact_detection(current_state, test_impact_threshold):
                impact_count += 1
                if impact_count > record_washout:
                    interval = (i - last_impact_step) * dt if last_impact_step is not None else 0.0
                    impact_intervals_local.append(interval)
                last_impact_step = i
                last_impact_idx = i

            time_since_impact = (i - last_impact_idx) * dt if last_impact_idx >= 0 else 0.0

            fill_esn_input(current_state, gamma, time_since_impact, 
                           current_state - last_impact_n, current_state - last_impact_p,
                           esn_input_buf)
            esn_state = local_esn.forward(esn_input_buf)

            curr_pred = esn_state @ coef_T  # linear readout

            if time_since_impact == 0.0:
                last_impact_n = current_state
                last_impact_p = curr_pred

            win_std.append(curr_pred)
            win_gap.append(curr_pred)
            test_prediction_list.append(curr_pred)

            # Stop criteria: static
            if (len(test_prediction_list) > static_threshold) and (len(win_std) == static_horizon):
                arr = np.asarray(win_std)
                if arr.std(axis=0).mean() < static_criteria:
                    break

            # Stop criteria: fluctuation
            if (len(test_prediction_list) > fluc_threshold) and (i > last_impact_idx + fluc_horizon + 10) \
               and (impact_count < min(fluc_threshold // 700, 10)):
                arr2 = np.asarray(win_gap)
                gap = arr2.max(axis=0) - arr2.min(axis=0)
                if gap[:2].min() > fluc_criteria:
                    break

            # Stop criteria: enough impacts
            if impact_count >= impact_threshold:
                if impact_intervals_local:
                    mean_interval = float(np.mean(impact_intervals_local))
                    std_interval  = float(np.std(impact_intervals_local))
                else:
                    mean_interval = None
                    std_interval  = None
                impact_intervals = impact_intervals_local
                reached = True
                reached_sim = sim_id
                break

        if reached:
            test_pred = np.array(test_prediction_list)
            break

    if reached:
        if print_results:
            print(f"run {notion} [gamma={gamma:.4f}] reached in sim {reached_sim} | "
                  f"mean={mean_interval:.3f}, std={std_interval:.3f}")
        if done_flags_dict is not None:
            done_flags_dict[gamma_key] = True
    if print_results:
        print(f"run {notion} [gamma={gamma:.4f}] not reached.")

    return {
        'gamma': float(gamma),
        'impact_intervals': impact_intervals if reached else [],
        'test_pred': test_pred,
        'reached': reached,
        'skipped': False,
    }