import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from scipy.interpolate import interp1d
import os, numpy as np
from importlib import reload
import esn_runner
reload(esn_runner)
from esn_runner import EchoStateNetwork

# --- turn on interactive / non-blocking display for loops ---
plt.ion()  # interactive mode: show() will not block


# ---------------------------------------------------------------------
# Utility: build ESN input by concatenating state and impact context
# ---------------------------------------------------------------------
def expand_state(state, gamma, time_since_impact, last_impact_n, last_impact_p):
    """
    Construct the (1, 14) ESN input vector from a 4D state and context.

    Args:
        state: array-like shape (4,) or (1,4) -> [theta_ns, theta_s, theta_dot_ns, theta_dot_s]
        gamma: scalar slope parameter (float)
        time_since_impact: seconds since last impact (float)
        last_impact_n: 4D state at last impact 'negative' side (np.ndarray)
        last_impact_p: 4D state at the time step after impact (np.ndarray)

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
        theta_ns, theta_s, theta_dot_ns, theta_dot_s,      # 4
        gamma,                                             # 1
        time_since_impact,                                 # 1
        theta_ns_n, theta_s_n, dot_ns_n, dot_s_n,          # 4
        theta_ns_p, theta_s_p, dot_ns_p, dot_s_p,          # 4
    ])
    return extended_in_state


# ---------------------------------------------------------------------
# Utility: impact (event) detector
# ---------------------------------------------------------------------
def impact_detection(state, threshold):
    """
    Detect impact event using geometric conditions.

    Args:
        state: array-like (4,) -> [theta_ns, theta_s, theta_dot_ns, theta_dot_s]
        threshold: float tolerance

    Returns:
        int: 1 if detected, otherwise 0
    """
    ns, s, dot_ns, dot_s = state
    if (abs(2 * ns - s) < threshold + 1e-6) and (ns < 0 + 1e-6) and (2 * dot_ns < dot_s + 1e-6):
        return 1
    else:
        return 0


# ---------------------------------------------------------------------
# Data: read files, resample to uniform dt, add noise, detect impacts
# ---------------------------------------------------------------------
def make_training_data(raw_paths, file_ids, noise_std, dt, train_impact_threshold, sampling_rate):
    """
    Build training samples from a subset of files.

    Args:
        raw_paths: list of filenames under base path
        file_ids: indices to choose from raw_paths
        noise_std: gaussian noise std added to time series
        dt: target sampling interval (s)
        train_impact_threshold: threshold for impact detection
        sampling_rate: Hz

    Returns:
        list[dict]: each with {'time_series': (T,4), 'impact_moments': list[int]}
    """
    all_train_data = []

    train_gamma = 0.01280
    base_path = f'./real_data/gamma={train_gamma:.6f}'
    for file_id in file_ids:
        file_path = os.path.join(base_path, raw_paths[file_id])

        # read 5 cols: time + 4 states
        df = pd.read_csv(file_path, delim_whitespace=True, usecols=range(5))
        df.columns = ["time", "theta_ns", "theta_s", "theta_dot_ns", "theta_dot_s"]
        df_unique = df.drop_duplicates(subset=["time"], keep="first")
        raw_data = df_unique.to_numpy()

        raw_timestamps, data = raw_data[:, 0], raw_data[:, 1:]

        # resample to uniform grid
        start_time, end_time = raw_timestamps[0], raw_timestamps[-1]
        timestamps = np.arange(start_time, end_time, dt)

        interp_func = lambda d: interp1d(raw_timestamps, d, kind='linear', fill_value="extrapolate")(timestamps)
        theta_ns = interp_func(data[:, 0])
        theta_s = interp_func(data[:, 1])
        theta_dot_ns = interp_func(data[:, 2])
        theta_dot_s = interp_func(data[:, 3])

        time_series = np.column_stack([theta_ns, theta_s, theta_dot_ns, theta_dot_s])

        # small noise for regularization
        time_series += np.random.normal(0, noise_std, time_series.shape)

        # detect impacts
        impact_moments = []
        for i in range(timestamps.shape[0]):
            if impact_detection(time_series[i], train_impact_threshold):
                impact_moments.append(i)

        # keep well-separated impacts (>= 2 * sampling_rate steps apart)
        filtered_impact_moments = []
        last_impact = impact_moments[0]
        for i in range(1, len(impact_moments)):
            if impact_moments[i] > last_impact + int(2 * sampling_rate):
                filtered_impact_moments.append(last_impact)
            last_impact = impact_moments[i]
        filtered_impact_moments.append(impact_moments[-1])

        # slice usable segment between first & last impact
        notion = filtered_impact_moments[0]
        time_series = time_series[filtered_impact_moments[0]:filtered_impact_moments[-1]]

        # re-index impacts to the sliced segment
        for i in range(len(filtered_impact_moments)):
            filtered_impact_moments[i] -= notion

        all_train_data.append({
            'time_series': time_series,
            'impact_moments': filtered_impact_moments
        })

    return all_train_data


# ---------------------------------------------------------------------
# Train two readouts (chaos & non-chaos) on different subsets
# ---------------------------------------------------------------------
def train_esn_model(chaos_data, nonchaos_data, esn_in_size, esn_r_size,
                    rho, leaky, dens, esn_in_scale, ridge_alpha, dt):
    """
    Train two linear readouts (Ridge) on top of a single ESN reservoir:
    one for 'chaos' subset, one for 'non-chaos' subset.

    Returns:
        esn: EchoStateNetwork (reservoir only)
        chaos_model: Ridge readout for chaos subset
        nonchaos_model: Ridge readout for non-chaos subset
    """
    washout = 100
    gamma = 0.01280
    esn = EchoStateNetwork(esn_in_size, esn_r_size, rho, leaky, dens, esn_in_scale)

    chaos_inputs, chaos_targets = [], []
    nonchaos_inputs, nonchaos_targets = [], []

    # ----- chaos subset -----
    for chaos_sample in chaos_data:
        time_series = chaos_sample['time_series']
        gt_impact_moments = chaos_sample['impact_moments']

        last_impact_idx = -1
        last_impact_n, last_impact_p = np.zeros(4), np.zeros(4)

        # washout
        for _ in range(washout):
            esn.forward(expand_state(np.zeros(4), gamma, 0., np.zeros(4), np.zeros(4)))

        train_esn_states = []

        # teacher-forcing (collect states -> next-step targets)
        for i in range(len(time_series)-1):
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
                last_impact_p = time_series[i+1]

        chaos_inputs.append(train_esn_states[washout:])
        chaos_targets.append(time_series[1+washout:])

    chaos_inputs = np.vstack(chaos_inputs)
    chaos_targets = np.vstack(chaos_targets)

    # ----- non-chaos subset -----
    for nonchaos_sample in nonchaos_data:
        time_series = nonchaos_sample['time_series']
        gt_impact_moments = nonchaos_sample['impact_moments']

        last_impact_idx = -1
        last_impact_n, last_impact_p = np.zeros(4), np.zeros(4)

        for _ in range(washout):
            esn.forward(expand_state(np.zeros(4), gamma, 0., np.zeros(4), np.zeros(4)))

        train_esn_states = []

        for i in range(len(time_series)-1):
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
                last_impact_p = time_series[i+1]

        nonchaos_inputs.append(train_esn_states[washout:])
        nonchaos_targets.append(time_series[1+washout:])

    nonchaos_inputs = np.vstack(nonchaos_inputs)
    nonchaos_targets = np.vstack(nonchaos_targets)

    # separate readouts
    chaos_model = Ridge(alpha=ridge_alpha, fit_intercept=False)
    nonchaos_model = Ridge(alpha=ridge_alpha, fit_intercept=False)

    chaos_model.fit(chaos_inputs, chaos_targets)
    nonchaos_model.fit(nonchaos_inputs, nonchaos_targets)

    # quick diagnostics
    chaos_pred = chaos_model.predict(chaos_inputs)
    nonchaos_pred = nonchaos_model.predict(nonchaos_inputs)
    chaos_error = chaos_targets - chaos_pred
    nonchaos_error = nonchaos_targets - nonchaos_pred

    print(f'Chaos train error: {((chaos_error**2).mean(axis=0)**0.5).mean()} \n'
          f'Non-chaos train error: {((nonchaos_error**2).mean(axis=0)**0.5).mean()} ')

    return esn, chaos_model, nonchaos_model


# ---------------------------------------------------------------------
# Rollout: teacher forcing for teacher_duration, then free-run prediction
# ---------------------------------------------------------------------
def run_esn_pred(single_teacher_data,
                 esn,
                 model,
                 test_impact_threshold,
                 teacher_duration,
                 pred_duration):
    """
    Roll out ESN:
      - teacher forcing for `teacher_duration` steps
      - then free-run for additional `pred_duration` steps
    """
    washout = 100
    gamma = 0.01280
    all_esn_pred = []

    time_series = single_teacher_data['time_series']
    gt_impact_moments = single_teacher_data['impact_moments']

    last_impact_idx = -1
    last_impact_n, last_impact_p = np.zeros(4), np.zeros(4)

    # reset (washout) with zeros
    for _ in range(washout):
        esn.forward(expand_state(np.zeros(4), gamma, 0., np.zeros(4), np.zeros(4)))

    # 1) teacher-forced segment
    for i in range(teacher_duration):
        current_state = time_series[i]

        if i in gt_impact_moments:
            last_impact_idx = i

        time_since_impact = (i - last_impact_idx) * dt if last_impact_idx >= 0 else 0
        esn_input = expand_state(current_state, gamma, time_since_impact,
                                 current_state - last_impact_n,
                                 current_state - last_impact_p)
        esn_state = esn.forward(esn_input)
        current_pred = model.predict(esn_state.reshape(1, -1))[0]
        all_esn_pred.append(current_pred)

        if i in gt_impact_moments:
            last_impact_n = time_series[i]
            last_impact_p = time_series[i+1]

    # 2) free-run prediction segment
    for i in range(teacher_duration, teacher_duration + pred_duration):
        current_state = all_esn_pred[-1]

        if impact_detection(current_state, test_impact_threshold):
            last_impact_idx = i

        time_since_impact = (i - last_impact_idx) * dt if last_impact_idx >= 0 else 0
        esn_input = expand_state(current_state, gamma, time_since_impact,
                                 current_state - last_impact_n,
                                 current_state - last_impact_p)
        esn_state = esn.forward(esn_input)
        current_pred = model.predict(esn_state.reshape(1, -1))[0]

        if time_since_impact == 0:
            last_impact_n = current_state
            last_impact_p = current_pred

        all_esn_pred.append(current_pred)

    return np.array(all_esn_pred)


# ===================== configuration =====================
sampling_rate = 100
dt = 1.0 / sampling_rate

esn_in_size = expand_state(np.zeros(4), 0, 0, np.zeros(4), np.zeros(4)).shape[1]
esn_r_size = 400
rho = 0.9
leaky = 0.1
dens = 1.
esn_in_scale = 1.1

train_gamma = 0.01280
data_location = f'./real_data/gamma={train_gamma:.6f}'

# load all txt file names under the gamma folder
raw_files = []
raw_paths = []
for filename in os.listdir(data_location):
    filepath = os.path.join(data_location, filename)
    if os.path.isfile(filepath) and filename.endswith('.txt'):
        file = np.loadtxt(filepath, delimiter=" ", skiprows=1)[:,:5]
        raw_files.append(file)
        raw_paths.append(filename)

# split indices by your rule
nonchaos = [0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 13]
chaos = [3, 7, 12, 14, 15]

washout = 100

train_impact_threshold = 1 / sampling_rate
test_impact_threshold = 1 / sampling_rate

noise_std = 3e-4

# build datasets
chaos_train_data = make_training_data(raw_paths, chaos, noise_std, dt, train_impact_threshold, sampling_rate)
nonchaos_train_data = make_training_data(raw_paths, nonchaos, noise_std, dt, train_impact_threshold, sampling_rate)

# train two readouts
ridge_alpha = 1e-8
esn, chaos_model, nonchaos_model = train_esn_model(
    chaos_train_data, nonchaos_train_data,
    esn_in_size, esn_r_size, rho, leaky, dens, esn_in_scale, ridge_alpha, dt
)

# thresholds for detecting “stuck”/“fluctuation” (not used in this plotting demo)
static_threshold = 2000
static_horizon = 200
static_criteria = 1e-2

fluc_threshold = 2000
fluc_horizon = 5
fluc_criteria = 1e-1

record_washout = 50
impact_threshold = record_washout + 50

test_duration = int(4.3 * impact_threshold * sampling_rate)
gamma = train_gamma

save_path = f'./multi_attractor/unseen_attractor/'
os.makedirs(save_path, exist_ok=True)

teacher_duration = 40
pred_duration = 80

# pick one readout & one dataset to roll out
pred_model = chaos_model          # or chaos_model
pred_data = nonchaos_train_data         # or nonchaos_train_data

# ===================== plotting loop =====================
for idx, teacher_sample in enumerate(pred_data):
    esn_pred = run_esn_pred(
        teacher_sample, esn, pred_model, test_impact_threshold,
        teacher_duration*sampling_rate, pred_duration*sampling_rate
    )

    # named access (not strictly needed)
    theta_ns, theta_s = esn_pred[:, 0], esn_pred[:, 1]
    theta_ns_dot, theta_s_dot = esn_pred[:, 2], esn_pred[:, 3]

    esn_teacher = esn_pred[washout:teacher_duration*sampling_rate, :]
    esn_stable  = esn_pred[(teacher_duration+50)*sampling_rate:, :]
    teacher_data = teacher_sample['time_series'][washout:, :]

    plt.figure(figsize=(5, 4))
    # ground truth (red)
    plt.plot(teacher_data[:, 0], teacher_data[:, 2], color='red')
    plt.plot(teacher_data[:, 1], teacher_data[:, 3], color='red')
    # teacher-forced segment (blue)
    plt.plot(esn_teacher[:, 0], esn_teacher[:, 2], color='blue')
    plt.plot(esn_teacher[:, 1], esn_teacher[:, 3], color='blue')
    # free-run segment (black)
    plt.plot(esn_stable[:, 0], esn_stable[:, 2], color='black')
    plt.plot(esn_stable[:, 1], esn_stable[:, 3], color='black')

    plt.xlabel('Angle (rad)')
    plt.ylabel('Angular velocity (rad/s)')
    plt.grid(True)

    # --- non-blocking show: will not wait for you to close the window ---
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.3)   # small pause so the figure can render before next loop
    # If you want to save and close instead:
    # plt.savefig(os.path.join(save_path, f'phase_{idx:03d}.png'), dpi=150)
    # plt.close()