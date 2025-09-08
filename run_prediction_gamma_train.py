import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from scipy.interpolate import interp1d

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from multiprocessing import Manager
import random, os, numpy as np

from joblib import dump, load
from importlib import reload
import esn_runner
reload(esn_runner)

from esn_runner import EchoStateNetwork, run_one_gamma_with_flag
import copy

from scipy.stats import wasserstein_distance
import argparse


def get_real_bifurcation():
    """
    Load the ground-truth bifurcation data.

    Returns:
        dict[float -> list[float]]: Mapping gamma -> list of impact intervals.
    """
    df = pd.read_csv("real_bifurcation.csv")
    grouped = df.groupby("gamma")["interval"].apply(list)
    return grouped.to_dict()


def expand_state(state, gamma, time_since_impact, last_impact_n, last_impact_p):
    """
    Build the ESN input by concatenating the 4D state and contextual scalars.

    Args:
        state (array-like): [theta_ns, theta_s, theta_dot_ns, theta_dot_s]
        gamma (float): slope parameter
        time_since_impact (float): seconds since last impact
        last_impact_n (np.ndarray): state at last impact (negative step)
        last_impact_p (np.ndarray): subsequent state after last impact (positive step)

    Returns:
        np.ndarray of shape (1, 14): extended input fed to ESN.
    """
    state = np.atleast_2d(state)

    theta_ns = state[:, 0]
    theta_s = state[:, 1]
    theta_dot_ns = state[:, 2]
    theta_dot_s = state[:, 3]

    theta_ns_n, theta_s_n, dot_ns_n, dot_s_n = last_impact_n
    theta_ns_p, theta_s_p, dot_ns_p, dot_s_p = last_impact_p

    extended_in_state = np.column_stack([
        theta_ns, theta_s, theta_dot_ns, theta_dot_s,      # 4 dims
        gamma,                                             # 1
        time_since_impact,                                 # 1
        theta_ns_n, theta_s_n, dot_ns_n, dot_s_n,          # 4 (pre-impact)
        theta_ns_p, theta_s_p, dot_ns_p, dot_s_p,          # 4 (post-impact)
    ])
    return extended_in_state


def impact_detection(state, threshold):
    """
    Event detector for 'impact' condition based on the model's geometry.

    Args:
        state (array-like): [theta_ns, theta_s, theta_dot_ns, theta_dot_s]
        threshold (float): detection tolerance

    Returns:
        int: 1 if impact is detected, else 0.
    """
    ns, s, dot_ns, dot_s = state
    if (abs(2 * ns - s) < threshold + 1e-6) and (ns < 0 + 1e-6) and (2 * dot_ns < dot_s + 1e-6):
        return 1
    else:
        return 0


def make_training_data(train_gammas, noise_std, dt, train_impact_threshold, sampling_rate):
    """
    Build training sequences from raw files for each gamma:
    - interpolate to uniform dt
    - add small Gaussian noise
    - detect & filter impacts; then window the usable segment

    Returns:
        list[dict]: Each item has keys:
            'gamma': float
            'time_series': np.ndarray (T, 4)
            'impact_moments': list[int] (indices in the sliced segment)
    """
    all_train_data = []

    for gamma in train_gammas:
        base_path = f"./real_data/gamma={gamma:.6f}/"
        txt_files = [f for f in os.listdir(base_path) if f.endswith(".txt")]

        for txt_file in txt_files:
            file_path = os.path.join(base_path, txt_file)

            # Load and deduplicate by time
            df = pd.read_csv(file_path, delim_whitespace=True, usecols=range(5))
            df.columns = ["time", "theta_ns", "theta_s", "theta_dot_ns", "theta_dot_s"]
            df_unique = df.drop_duplicates(subset=["time"], keep="first")
            raw_data = df_unique.to_numpy()

            raw_timestamps, data = raw_data[:, 0], raw_data[:, 1:]

            # Resample to uniform grid
            start_time, end_time = raw_timestamps[0], raw_timestamps[-1]
            timestamps = np.arange(start_time, end_time, dt)

            interp_func = lambda d: interp1d(raw_timestamps, d, kind='linear', fill_value="extrapolate")(timestamps)
            theta_ns = interp_func(data[:, 0])
            theta_s = interp_func(data[:, 1])
            theta_dot_ns = interp_func(data[:, 2])
            theta_dot_s = interp_func(data[:, 3])

            time_series = np.column_stack([theta_ns, theta_s, theta_dot_ns, theta_dot_s])

            # Add small noise for regularization
            time_series += np.random.normal(0, noise_std, time_series.shape)

            # Detect impacts on the regular grid
            impact_moments = []
            for i in range(timestamps.shape[0]):
                if impact_detection(time_series[i], train_impact_threshold):
                    impact_moments.append(i)

            # Keep impacts separated by at least 2 / sampling_rate seconds
            filtered_impact_moments = []
            last_impact = impact_moments[0]
            for i in range(1, len(impact_moments)):
                if impact_moments[i] > last_impact + int(2 * sampling_rate):
                    filtered_impact_moments.append(last_impact)
                last_impact = impact_moments[i]
            filtered_impact_moments.append(impact_moments[-1])

            # Slice usable segment between first and last filtered impacts
            notion = filtered_impact_moments[0]
            time_series = time_series[filtered_impact_moments[0]:filtered_impact_moments[-1]]

            # Shift impact indices into the sliced segment's coordinates
            for i in range(len(filtered_impact_moments)):
                filtered_impact_moments[i] -= notion

            all_train_data.append({
                'gamma': gamma,
                'time_series': time_series,
                'impact_moments': filtered_impact_moments
            })

    return all_train_data


def train_esn_model(
    train_gammas, noise_std, esn_in_size, esn_r_size,
    rho, leaky, dens, esn_in_scale, ridge_alpha, dt,
    train_impact_threshold, sampling_rate, washout
):
    """
    Train an ESN + linear readout (Ridge) on concatenated training sequences.

    Returns:
        all_train_data (list[dict]): preprocessed sequences for later init
        esn (EchoStateNetwork): trained reservoir (state dynamics only)
        model (Ridge): linear readout mapping reservoir -> next 4D state
        rmse_mean (float): mean RMSE across 4 dimensions on training data
    """
    all_train_data = make_training_data(
        train_gammas, noise_std, dt, train_impact_threshold, sampling_rate
    )
    esn = EchoStateNetwork(esn_in_size, esn_r_size, rho, leaky, dens, esn_in_scale)

    all_input = []
    all_target = []

    for idx, train_sample in enumerate(all_train_data):
        gamma = train_sample['gamma']
        time_series = train_sample['time_series']
        gt_impact_moments = train_sample['impact_moments']

        last_impact_idx = -1
        last_impact_n, last_impact_p = np.zeros(4), np.zeros(4)

        # ESN washout to settle internal state
        for _ in range(washout):
            esn.forward(expand_state(np.zeros(4), gamma, 0., np.zeros(4), np.zeros(4)))

        train_esn_states = []

        # Teacher-forced rollout to collect reservoir states and targets
        for i in range(len(time_series) - 1):
            current_state = time_series[i]

            if i in gt_impact_moments:
                last_impact_idx = i

            time_since_impact = (i - last_impact_idx) * dt if last_impact_idx >= 0 else 0
            esn_input = expand_state(
                current_state, gamma, time_since_impact,
                current_state - last_impact_n,
                current_state - last_impact_p
            )
            esn_state = esn.forward(esn_input)
            train_esn_states.append(esn_state)

            # Update “impact context” when crossing an impact
            if i in gt_impact_moments:
                last_impact_n = time_series[i]
                last_impact_p = time_series[i + 1]

        # Discard first `washout` states, align targets with next-step ground truth
        all_input.append(train_esn_states[washout:])
        all_target.append(time_series[1 + washout:])

    # Stack all sequences
    train_input = np.vstack(all_input)
    train_target = np.vstack(all_target)

    # Linear readout (no intercept)
    model = Ridge(alpha=ridge_alpha, fit_intercept=False)
    model.fit(train_input, train_target)

    # Train error (RMSE)
    train_pred = model.predict(train_input)
    train_error = train_target - train_pred
    rmse_mean = ((train_error ** 2).mean(axis=0) ** 0.5).mean()

    return all_train_data, esn, model, rmse_mean


def key5(x):
    """Format a float gamma to 5 decimal places and back to float (stable dict key)."""
    return float(f"{x:.5f}")


def solve_gammas_first_success(
    gamma_list, m,
    N,
    static_threshold, static_horizon, static_criteria,
    fluc_threshold, fluc_horizon, fluc_criteria,
    record_washout, impact_threshold,
    sampling_rate, dt,
    test_impact_threshold,
    esn, model, all_train_data,
    washout=100,
    base_seed=42,
    n_jobs=None,
    start_inflight_per_gamma=1,
    max_inflight_cap=8,
    shuffle_seed=None,
    print_results=True,
):
    """
    For each gamma in gamma_list, launch up to `m` independent trials in parallel.
    Return the first success per gamma (i.e., simulation that reaches enough impacts).
    Remaining trials of that gamma are cancelled.

    Concurrency:
        - Global pool of size n_jobs
        - Per-gamma in-flight cap grows as fewer gammas remain
        - Shared Manager dict used to signal 'done' per gamma
    """
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 4) - 1)

    # Randomize gamma order for better load balancing
    shuffled = list(gamma_list)
    (random.Random(shuffle_seed).shuffle(shuffled)
     if shuffle_seed is not None else random.shuffle(shuffled))

    manager = Manager()
    done_flags = manager.dict({key5(g): False for g in shuffled})

    results_by_gamma = {key5(g): None for g in shuffled}
    submitted_counts = {key5(g): 0 for g in shuffled}
    running_counts   = {key5(g): 0 for g in shuffled}
    orig_gamma       = {key5(g): g for g in shuffled}
    keys = list(orig_gamma.keys())

    # Track in-flight futures per gamma to cancel them on first success
    futures_by_key = {k: set() for k in keys}

    def remaining_keys():
        return [k for k in keys if (not done_flags[k]) and (submitted_counts[k] < m)]

    def current_per_key_cap():
        """Adaptive per-gamma cap based on number of remaining gammas."""
        rem = len(remaining_keys())
        if rem == 0:
            return 0
        cap = max(1, n_jobs // rem)
        return min(cap, max_inflight_cap)

    def can_submit(k):
        if done_flags[k]:
            return False
        if submitted_counts[k] >= m:
            return False
        return running_counts[k] < current_per_key_cap()

    def submit_one(ex, k):
        """Submit a new trial for a specific gamma key."""
        seed = base_seed + submitted_counts[k]
        fut = ex.submit(
            run_one_gamma_with_flag,
            submitted_counts[k] + 1, orig_gamma[k], N,
            static_threshold, static_horizon, static_criteria,
            fluc_threshold, fluc_horizon, fluc_criteria,
            record_washout, impact_threshold,
            sampling_rate, dt,
            test_impact_threshold,
            esn, model, all_train_data,
            done_flags, washout, seed, print_results,
        )
        submitted_counts[k] += 1
        running_counts[k]   += 1
        futures_by_key[k].add(fut)
        future_to_key[fut] = k

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        future_to_key = {}

        # Seed each gamma with up to `start_inflight_per_gamma` in-flight futures
        for g in shuffled:
            k = key5(g)
            n0 = min(start_inflight_per_gamma, m - submitted_counts[k])
            for _ in range(n0):
                submit_one(ex, k)

        # Fill idle workers greedily under current caps
        def fill_idle():
            while len(future_to_key) < n_jobs:
                cands = [k for k in keys if can_submit(k)]
                if not cands:
                    break
                random.shuffle(cands)
                submitted = False
                for k in cands:
                    if can_submit(k):
                        submit_one(ex, k)
                        submitted = True
                        if len(future_to_key) >= n_jobs:
                            break
                if not submitted:
                    break

        fill_idle()

        # Main loop: process completed futures and keep pool full
        while future_to_key:
            done, _ = wait(set(future_to_key.keys()), return_when=FIRST_COMPLETED)

            for fut in done:
                k_done = future_to_key.pop(fut)
                futures_by_key[k_done].discard(fut)
                running_counts[k_done] = max(0, running_counts[k_done]-1)

                r = fut.result()

                # On first success for this gamma: record, mark done, cancel others
                if r['reached'] and results_by_gamma[k_done] is None:
                    results_by_gamma[k_done] = r
                    done_flags[k_done] = True

                    for other in list(futures_by_key[k_done]):
                        if not other.done():
                            other.cancel()
                        futures_by_key[k_done].discard(other)

                # Refill idle slots (cap may change as #remaining changes)
                fill_idle()

        # Ensure each gamma has a result dict even if none succeeded
        for k in list(results_by_gamma.keys()):
            if results_by_gamma[k] is None:
                results_by_gamma[k] = {
                    'gamma': orig_gamma[k],
                    'impact_intervals': [],
                    'test_pred': None,
                    'reached': False,
                    'skipped': False,
                }

    return results_by_gamma


def save_single_result(test_pred_save_path, trial, single_result):
    """
    Persist one trial's results to:
        {test_pred_save_path}/{trial}/{gamma_dir}/{key}.csv

    Only saves array-like values; skips booleans/strings/meta fields.
    """
    trial_path = f'{test_pred_save_path}/{trial}'
    os.makedirs(trial_path, exist_ok=True)

    for save_gamma, result_dict in single_result.items():
        gamma_dir = f'{save_gamma:.5f}'
        gamma_path = f'{trial_path}/{gamma_dir}'
        os.makedirs(gamma_path, exist_ok=True)

        for key, val in result_dict.items():
            if key in ('reached', 'skipped', 'gamma') or isinstance(val, (bool, str)):
                continue

            arr = np.array(val)
            if arr.ndim == 0:
                arr = arr.reshape(1)

            np.savetxt(f'{gamma_path}/{key}.csv', arr, delimiter=',', comments='')

    print(f'Saving results for trial {trial}')


def calculate_wasserstein_distance(pred_dict):
    """
    Compare predicted impact-interval distributions against the real (per gamma)
    using the 1-Wasserstein distance.

    Args:
        pred_dict (dict): gamma -> {'reached': bool, 'impact_intervals': list[float], ...}

    Returns:
        dict[float -> float]: gamma -> W1 distance
    """
    out = {}
    real_dict = get_real_bifurcation()
    for g, real_vals in real_dict.items():
        if not pred_dict[g]['reached']:
            out[g] = 1e6  # penalize missing predictions

        r = np.asarray(real_vals, dtype=float)
        p = np.asarray(pred_dict[g]['impact_intervals'], dtype=float)
        w1 = float(wasserstein_distance(r, p))
        out[g] = w1

    return out


def main():
    """
    CLI entry:
      --gamma <float>   : train on this gamma plus two fixed ones, then evaluate across a grid
      --workers <int>   : number of processes for parallel evaluation
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gamma", type=float, required=True,
        help="Please assign gamma_train_pick value"
    )
    parser.add_argument(
        "--workers", type=int, required=True,
        help="Please assign number of workers"
    )
    args = parser.parse_args()

    gamma_train_pick = args.gamma
    workers = args.workers

    # Training set (one chosen + two anchors)
    train_gammas = [gamma_train_pick, 0.01801, 0.01901]

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

    # Gamma grid to evaluate
    gamma_predict = np.arange(0.01001, 0.01951, 0.0005).tolist()

    # Per-gamma solver parameters
    N = 30     # per-gamma trials
    m = 40     # max submissions per gamma (upper bound on attempts)

    # Stop criteria (static / fluctuation / impact count)
    static_threshold = 2000
    static_horizon = 200
    static_criteria = 1e-2

    fluc_threshold = 3000
    fluc_horizon = 5
    fluc_criteria = 1e-1

    record_washout = 50
    impact_threshold = record_washout + 50

    # Experiment-level loop: try up to `max_trials` to collect `trial_num` good trials
    trial_num = 30
    max_trials = 200

    test_pred_save_path = f'./train_gamma_results/{gamma_train_pick}'
    os.makedirs(test_pred_save_path, exist_ok=True)

    saved_results = 0
    tried_times = 0

    w_dists, ids, models = [], [], {}

    # Keep training ESN until training RMSE is close to noise level, then evaluate
    while (saved_results < trial_num) and (tried_times < max_trials):

        train_error = np.inf
        all_train_data, esn, model = None, None, None

        # Simple quality gate on training fit
        while train_error > 1.05 * noise_std:
            all_train_data, esn, model, train_error = train_esn_model(
                train_gammas, noise_std, esn_in_size, esn_r_size,
                rho, leaky, dens, esn_in_scale, ridge_alpha, dt,
                train_impact_threshold, sampling_rate, washout
            )

        # Evaluate across gamma grid, keep first success per gamma
        trial = saved_results + 1
        single_result = solve_gammas_first_success(
            gamma_list=gamma_predict,
            m=m, N=N,
            static_threshold=static_threshold, static_horizon=static_horizon, static_criteria=static_criteria,
            fluc_threshold=fluc_threshold, fluc_horizon=fluc_horizon, fluc_criteria=fluc_criteria,
            record_washout=record_washout, impact_threshold=impact_threshold,
            sampling_rate=sampling_rate, dt=dt, test_impact_threshold=test_impact_threshold,
            esn=esn, model=model, all_train_data=all_train_data,
            washout=washout, base_seed=1234 + trial * 100, n_jobs=workers, print_results=False,
        )

        # Accept a trial only if all gammas reached the target impacts
        achieved = all(sr['reached'] for sr in single_result.values())

        if achieved:
            save_single_result(test_pred_save_path, trial, single_result)
            saved_results += 1

            # Score with Wasserstein distance against the real bifurcation
            wasser_dist = calculate_wasserstein_distance(single_result)
            mean_w = np.mean(list(wasser_dist.values()))
            w_dists.append(mean_w)
            ids.append(saved_results)

            # Keep a copy of the model for later "best" selection
            models[saved_results] = {'esn': copy.deepcopy(esn), 'model': copy.deepcopy(model)}

        tried_times += 1

    # Choose best model by the smallest mean Wasserstein distance
    pairs = list(zip(w_dists, ids))
    pairs_sorted = sorted(pairs, key=lambda x: x[0])
    best_id = pairs_sorted[0][1]

    model_save_path = f'{test_pred_save_path}/best_model.joblib'
    dump({"esn": models[best_id]["esn"], "model": models[best_id]["model"]},
         model_save_path, compress=3)
    print("saved ->", model_save_path)


if __name__ == "__main__":
    main()