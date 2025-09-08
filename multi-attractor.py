import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from scipy.interpolate import interp1d
import os, numpy as np
from importlib import reload
import esn_runner
reload(esn_runner)
from esn_runner import EchoStateNetwork

# ---- make matplotlib non-blocking for looped plotting ----
plt.ion()


# ---------------------------------------------------------------------
# Build ESN input by concatenating state and impact context (shape: (1,14))
# ---------------------------------------------------------------------
def expand_state(state, gamma, time_since_impact, last_impact_n, last_impact_p):
    """
    Construct the ESN input vector from a 4D state and context.

    Args:
        state: array-like (4,) or (1,4) -> [theta_ns, theta_s, theta_dot_ns, theta_dot_s]
        gamma: float, slope parameter
        time_since_impact: float, seconds since last impact
        last_impact_n: (4,) state at impact time (pre)
        last_impact_p: (4,) state at next step after impact (post)

    Returns:
        np.ndarray of shape (1, 14)
    """
    state = np.atleast_2d(state)

    theta_ns = state[:, 0]
    theta_s = state[:, 1]
    theta_dot_ns = state[:, 2]
    theta_dot_s = state[:, 3]

    theta_ns_n, theta_s_n, dot_ns_n, dot_s_n = last_impact_n
    theta_ns_p, theta_s_p, dot_ns_p, dot_s_p = last_impact_p

    extended_in_state = np.column_stack([
        theta_ns, theta_s, theta_dot_ns, theta_dot_s,  # 4
        gamma,                                         # 1
        time_since_impact,                             # 1
        theta_ns_n, theta_s_n, dot_ns_n, dot_s_n,      # 4
        theta_ns_p, theta_s_p, dot_ns_p, dot_s_p,      # 4
    ])
    return extended_in_state


# ---------------------------------------------------------------------
# Impact detector with geometric conditions
# ---------------------------------------------------------------------
def impact_detection(state, threshold):
    """
    Detect impact event.

    Args:
        state: array-like (4,). Order: [theta_ns, theta_s, theta_dot_ns, theta_dot_s]
        threshold: float, tolerance

    Returns:
        int: 1 if detected, 0 otherwise
    """
    ns, s, dot_ns, dot_s = state
    if (abs(2 * ns - s) < threshold + 1e-6) and (ns < 0 + 1e-6) and (2 * dot_ns < dot_s + 1e-6):
        return 1
    else:
        return 0


# ---------------------------------------------------------------------
# Load & resample data to uniform dt; add noise; detect & filter impacts
# ---------------------------------------------------------------------
def make_training_data(train_gammas, noise_std, dt, train_impact_threshold, sampling_rate):
    """
    Build training samples from multiple gamma folders.

    Returns:
        list of dicts:
          {
            'gamma': float,
            'time_series': (T,4),
            'impact_moments': list[int]  # indices in the (sliced) time_series
          }
    """
    all_train_data = []

    for gamma in train_gammas:
        base_path = f"./real_data/gamma={gamma:.6f}/"
        txt_files = [f for f in os.listdir(base_path) if f.endswith(".txt")]

        for txt_file in txt_files:
            file_path = os.path.join(base_path, txt_file)

            # Read 5 cols: time + 4 states
            df = pd.read_csv(file_path, delim_whitespace=True, usecols=range(5))
            df.columns = ["time", "theta_ns", "theta_s", "theta_dot_ns", "theta_dot_s"]
            df_unique = df.drop_duplicates(subset=["time"], keep="first")
            raw_data = df_unique.to_numpy()

            raw_timestamps, data = raw_data[:, 0], raw_data[:, 1:]

            # Resample to uniform timeline
            start_time, end_time = raw_timestamps[0], raw_timestamps[-1]
            timestamps = np.arange(start_time, end_time, dt)

            interp_func = lambda d: interp1d(raw_timestamps, d, kind='linear', fill_value="extrapolate")(timestamps)
            theta_ns = interp_func(data[:, 0])
            theta_s = interp_func(data[:, 1])
            theta_dot_ns = interp_func(data[:, 2])
            theta_dot_s = interp_func(data[:, 3])

            time_series = np.column_stack([theta_ns, theta_s, theta_dot_ns, theta_dot_s])

            # Mild noise for regularization/robustness
            time_series += np.random.normal(0, noise_std, time_series.shape)

            # Detect impacts on the resampled series
            impact_moments = []
            for i in range(timestamps.shape[0]):
                if impact_detection(time_series[i], train_impact_threshold):
                    impact_moments.append(i)

            # Keep only well-separated impacts (>= 2 * sampling_rate apart)
            filtered_impact_moments = []
            last_impact = impact_moments[0]
            for i in range(1, len(impact_moments)):
                if impact_moments[i] > last_impact + int(2 * sampling_rate):
                    filtered_impact_moments.append(last_impact)
                last_impact = impact_moments[i]
            filtered_impact_moments.append(impact_moments[-1])

            # Slice the interval between the first and last impact
            base = filtered_impact_moments[0]
            time_series = time_series[filtered_impact_moments[0]:filtered_impact_moments[-1]]

            # Reindex impact indices to the sliced segment
            for i in range(len(filtered_impact_moments)):
                filtered_impact_moments[i] -= base

            all_train_data.append({
                'gamma': gamma,
                'time_series': time_series,
                'impact_moments': filtered_impact_moments
            })

    return all_train_data


# ---------------------------------------------------------------------
# Train a single Ridge readout on top of one ESN (all gammas pooled)
# ---------------------------------------------------------------------
def train_esn_model(train_gammas, noise_std, esn_in_size, esn_r_size,
                    rho, leaky, dens, esn_in_scale, ridge_alpha, dt,
                    train_impact_threshold, sampling_rate):
    """
    Train a single Ridge readout on top of one ESN, using all selected gammas.

    Returns:
        all_train_data (list of dicts), esn (EchoStateNetwork), model (Ridge),
        mean_RMSE (float, averaged over dims)
    """
    washout = 100

    all_train_data = make_training_data(
        train_gammas, noise_std, dt, train_impact_threshold, sampling_rate
    )
    esn = EchoStateNetwork(esn_in_size, esn_r_size, rho, leaky, dens, esn_in_scale)

    all_input = []
    all_target = []

    for train_sample in all_train_data:
        gamma = train_sample['gamma']
        time_series = train_sample['time_series']
        gt_impact_moments = train_sample['impact_moments']

        # Impact context
        last_impact_idx = -1
        last_impact_n, last_impact_p = np.zeros(4), np.zeros(4)

        # Washout with zeros
        for _ in range(washout):
            esn.forward(expand_state(np.zeros(4), gamma, 0., np.zeros(4), np.zeros(4)))

        # Collect ESN states (teacher-forced) and next-step targets
        train_esn_states = []
        for i in range(len(time_series) - 1):
            current_state = time_series[i]

            if i in gt_impact_moments:
                last_impact_idx = i

            time_since_impact = (i - last_impact_idx) * dt if last_impact_idx >= 0 else 0
            esn_input = expand_state(current_state, gamma, time_since_impact,
                                     current_state - last_impact_n,
                                     current_state - last_impact_p)
            esn_state = esn.forward(esn_input)
            train_esn_states.append(esn_state)

            if i in gt_impact_moments:
                last_impact_n = time_series[i]
                last_impact_p = time_series[i + 1]

        all_input.append(train_esn_states[washout:])
        all_target.append(time_series[1 + washout:])

    train_input = np.vstack(all_input)   # (N_s, reservoir)
    train_target = np.vstack(all_target) # (N_s, 4)

    model = Ridge(alpha=ridge_alpha, fit_intercept=False)
    model.fit(train_input, train_target)

    # Diagnostics
    train_pred = model.predict(train_input)
    train_error = train_target - train_pred
    per_dim_rmse = (train_error ** 2).mean(axis=0) ** 0.5
    mean_rmse = per_dim_rmse.mean()

    print(
        f'Train RMSE per dim: {per_dim_rmse}; '
        f'Mean: {mean_rmse}\n'
    )

    return all_train_data, esn, model, mean_rmse


# ===================== Config & Training =====================
np.random.seed(12345)

gamma_train_pick = 0.01280
train_gammas = [gamma_train_pick]

sampling_rate = 100
dt = 1.0 / sampling_rate

esn_in_size = expand_state(np.zeros(4), 0, 0, np.zeros(4), np.zeros(4)).shape[1]
esn_r_size = 400
rho = 0.9
leaky = 0.1
dens = 1.
esn_in_scale = 1.1

washout = 100
train_impact_threshold = 1 / sampling_rate
test_impact_threshold = 1 / sampling_rate
noise_std = 3e-4
ridge_alpha = 1e-8

# Keep retraining until RMSE is reasonably small (relative to noise)
train_error = np.inf
all_train_data, esn, model = None, None, None
while train_error > 1.05 * noise_std:
    all_train_data, esn, model, train_error = train_esn_model(
        train_gammas, noise_std, esn_in_size, esn_r_size,
        rho, leaky, dens, esn_in_scale, ridge_alpha, dt,
        train_impact_threshold, sampling_rate
    )
print(f'Training completed with mean RMSE: {train_error:.6g}')


# ===================== Free-roll Simulations =====================
N = 1000

# “stuck” / “fluctuation” heuristics (used to early-stop)
static_threshold = 2000
static_horizon = 200
static_criteria = 1e-2

fluc_threshold = 2000
fluc_horizon = 5
fluc_criteria = 1e-1

record_washout = 50
impact_threshold = record_washout + 50

# total time budget in steps
test_duration = int(4.3 * impact_threshold * sampling_rate)
gamma = gamma_train_pick

output_results = {}

for sim_id in range(N):
    print(f"Running Simulation {sim_id + 1}/{N} for gamma={gamma:.5f}...")

    # init state ~ N(0, 0.1^2)
    init_state = np.random.normal(0, 0.1, 4)
    test_prediction_list = [init_state]
    impact_moments = []

    # reset ESN internal state with a washout of zeros
    for _ in range(washout):
        esn.forward(expand_state(np.zeros(4), gamma, 0., np.zeros(4), np.zeros(4)))

    impact_count = 0
    last_impact_idx = -1
    last_impact_n, last_impact_p = np.zeros(4), np.zeros(4)

    for i in range(test_duration):
        current_state = test_prediction_list[-1]

        # detect impact
        if impact_detection(current_state, test_impact_threshold):
            last_impact_idx = i
            impact_moments.append(i)
            impact_count += 1

        # time since last impact (seconds)
        time_since_impact = (i - last_impact_idx) * dt if last_impact_idx >= 0 else 0.0

        # forward ESN + linear readout
        esn_input = expand_state(
            current_state, gamma, time_since_impact,
            current_state - last_impact_n,
            current_state - last_impact_p
        )
        esn_state = esn.forward(esn_input)
        current_pred = model.predict(esn_state.reshape(1, -1))[0]

        # refresh “pre/post impact” references exactly at impact time
        if time_since_impact == 0.0:
            last_impact_n = current_state
            last_impact_p = current_pred

        test_prediction_list.append(current_pred)

        # early stop: nearly static (small recent std)
        if len(test_prediction_list) > static_threshold:
            recent_std = np.std(test_prediction_list[-static_horizon:], axis=0).mean()
            if recent_std < static_criteria:
                print(f"Simulation became static at step {i}, impact_count: {impact_count}.")
                break

        # early stop: strong fluctuation (large recent std)
        if (len(test_prediction_list) > fluc_threshold) and (i > last_impact_idx + fluc_horizon + 10):
            recent_std = np.std(test_prediction_list[-fluc_horizon:], axis=0).mean()
            if recent_std > fluc_criteria:
                print(f"Simulation became fluctuating at step {i}, impact_count: {impact_count}.")
                break

        # reached desired number of impacts -> record and visualize
        if impact_count >= impact_threshold:
            print(f"[gamma={gamma:.5f}] Simulation {sim_id + 1}/{N} reached {impact_threshold} impacts.")
            test_pred = np.array(test_prediction_list)

            output_results[sim_id] = {
                'init_state': init_state,
                'test_pred': test_pred,
                'impact_moments': impact_moments,
            }

            # Non-blocking scatter of two phase portraits
            plt.figure(figsize=(5, 4))
            plt.plot(test_pred[-5000:, 0], test_pred[-5000:, 2])
            plt.plot(test_pred[-5000:, 1], test_pred[-5000:, 3])
            plt.xlabel('Angle (rad)')
            plt.ylabel('Angular velocity (rad/s)')
            # show init_state compactly
            init_str = ", ".join(f"{v:.6f}" for v in init_state)
            plt.title(f'Init: [{init_str}]')
            plt.grid(True, alpha=0.3)

            # show without blocking the loop
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.3)

            break