# imperfect.py
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from scipy.interpolate import interp1d

from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from importlib import reload

# --- external: your ESN runner module ---
import esn_runner
reload(esn_runner)
from esn_runner import EchoStateNetwork, run_one_gamma_with_flag


# ------------------------------
# Utilities
# ------------------------------
def get_real_bifurcation():
    """Load ground-truth bifurcation intervals from CSV.
    Returns:
        dict: {gamma_float: [interval, ...], ...}
    """
    df = pd.read_csv("real_bifurcation.csv")
    grouped = df.groupby("gamma")["interval"].apply(list)
    return grouped.to_dict()


def expand_state(state, gamma, time_since_impact, last_impact_n, last_impact_p):
    """Build the ESN input vector from 4D state + context scalars/vectors.

    Args:
        state: shape (4,) or (1,4)
        gamma: float
        time_since_impact: float
        last_impact_n: shape (4,)
        last_impact_p: shape (4,)

    Returns:
        ndarray shape (1, 14)
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
        theta_ns_n, theta_s_n, dot_ns_n, dot_s_n,
        theta_ns_p, theta_s_p, dot_ns_p, dot_s_p,
    ])
    return extended_in_state


def impact_detection(state, threshold):
    """Impact condition test on a 4D state."""
    ns, s, dot_ns, dot_s = state
    # if imperfect touch down impact detection, using the following logic judgement:
    if (abs(2.01 * ns - 0.99 * s) < threshold + 1e-6) and (ns < -0.05 + 1e-6) and (1.95 * dot_ns < dot_s + 1e-6):
        return 1
    else:
        return 0


def plot_bifurcation(gammas_pred, intervals_pred):
    """Scatter plot: real vs predicted impact intervals over gamma."""
    real_dict = get_real_bifurcation()
    sorted_gammas = np.array(sorted(real_dict.keys()))

    plt.figure(figsize=(8, 4))
    # real
    for gamma in sorted_gammas:
        intervals_real = real_dict[gamma]
        plt.scatter(
            [gamma] * len(intervals_real),
            intervals_real,
            color="red",
            alpha=0.5,
            label="Real" if gamma == sorted_gammas[0] else "",
        )
    # predicted
    plt.scatter(gammas_pred, intervals_pred, alpha=0.5, label="Predicted")

    plt.xlabel("Gamma")
    plt.ylabel("Impact Interval (s)")
    plt.xlim(0.009, 0.020)
    plt.ylim(3.5, 4.2)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def make_training_data(train_gammas, noise_std, dt, train_impact_threshold, sampling_rate):
    """Load and interpolate training trajectories from multiple gammas.

    Returns:
        list of dicts:
            {'gamma': float, 'time_series': (T,4), 'impact_moments': list[int]}
    """
    all_train_data = []

    for gamma in train_gammas:
        base_path = f"../results/gamma={gamma:.6f}/"
        txt_files = [f for f in os.listdir(base_path) if f.endswith(".txt")]

        for txt_file in txt_files:
            file_path = os.path.join(base_path, txt_file)

            df = pd.read_csv(file_path, sep=r"\s+", usecols=range(5))
            df.columns = ["time", "theta_ns", "theta_s", "theta_dot_ns", "theta_dot_s"]
            df_unique = df.drop_duplicates(subset=["time"], keep="first")
            raw_data = df_unique.to_numpy()

            raw_timestamps, data = raw_data[:, 0], raw_data[:, 1:]

            start_time, end_time = raw_timestamps[0], raw_timestamps[-1]
            timestamps = np.arange(start_time, end_time, dt)

            interp_func = lambda d: interp1d(
                raw_timestamps, d, kind="linear", fill_value="extrapolate"
            )(timestamps)
            theta_ns = interp_func(data[:, 0])
            theta_s = interp_func(data[:, 1])
            theta_dot_ns = interp_func(data[:, 2])
            theta_dot_s = interp_func(data[:, 3])

            time_series = np.column_stack([theta_ns, theta_s, theta_dot_ns, theta_dot_s])
            time_series += np.random.normal(0, noise_std, time_series.shape)

            # detect impacts
            impact_moments = []
            for i in range(timestamps.shape[0]):
                if impact_detection(time_series[i], train_impact_threshold):
                    impact_moments.append(i)

            # filter impacts at least 2*sampling_rate apart
            filtered_impact_moments = []
            if len(impact_moments) > 0:
                last_impact = impact_moments[0]
                for i in range(1, len(impact_moments)):
                    if impact_moments[i] > last_impact + int(2 * sampling_rate):
                        filtered_impact_moments.append(last_impact)
                    last_impact = impact_moments[i]
                filtered_impact_moments.append(impact_moments[-1])

                # normalize window to [first,last] impact
                notion = filtered_impact_moments[0]
                time_series = time_series[filtered_impact_moments[0] : filtered_impact_moments[-1]]
                for i in range(len(filtered_impact_moments)):
                    filtered_impact_moments[i] -= notion
            else:
                filtered_impact_moments = []

            all_train_data.append(
                {
                    "gamma": gamma,
                    "time_series": time_series,
                    "impact_moments": filtered_impact_moments,
                }
            )

    return all_train_data


def train_esn_model(
    train_gammas,
    noise_std,
    esn_in_size,
    esn_r_size,
    rho,
    leaky,
    dens,
    esn_in_scale,
    ridge_alpha,
    dt,
    train_impact_threshold,
    sampling_rate,
    washout,
):
    """Prepare training states for the ESN and fit a ridge readout."""
    all_train_data = make_training_data(
        train_gammas, noise_std, dt, train_impact_threshold, sampling_rate
    )
    esn = EchoStateNetwork(esn_in_size, esn_r_size, rho, leaky, dens, esn_in_scale)

    all_input = []
    all_target = []

    for train_sample in all_train_data:
        gamma = train_sample["gamma"]
        time_series = train_sample["time_series"]
        gt_impact_moments = set(train_sample["impact_moments"])

        last_impact_idx = -1
        last_impact_n, last_impact_p = np.zeros(4), np.zeros(4)

        # washout the internal state with zeros
        for _ in range(washout):
            esn.forward(expand_state(np.zeros(4), gamma, 0.0, np.zeros(4), np.zeros(4)))

        train_esn_states = []

        for i in range(len(time_series) - 1):
            current_state = time_series[i]
            if i in gt_impact_moments:
                last_impact_idx = i

            time_since_impact = (i - last_impact_idx) * dt if last_impact_idx >= 0 else 0.0
            esn_input = expand_state(
                current_state,
                gamma,
                time_since_impact,
                current_state - last_impact_n,
                current_state - last_impact_p,
            )
            esn_state = esn.forward(esn_input)
            train_esn_states.append(esn_state)

            if i in gt_impact_moments:
                last_impact_n = time_series[i]
                last_impact_p = time_series[i + 1]

        if len(train_esn_states) > washout:
            all_input.append(train_esn_states[washout:])
            all_target.append(time_series[1 + washout :])

    train_input = np.vstack(all_input)
    train_target = np.vstack(all_target)

    model = Ridge(alpha=ridge_alpha, fit_intercept=False)
    model.fit(train_input, train_target)

    train_pred = model.predict(train_input)
    train_error = train_target - train_pred

    print(
        f"Train error: {(train_error**2).mean(axis=0)**0.5}; "
        f"Mean: {((train_error**2).mean(axis=0)**0.5).mean()} \n"
    )

    return all_train_data, esn, model, ((train_error**2).mean(axis=0)**0.5).mean()


def key5(x):
    """Format a float gamma to fixed-precision float key."""
    return float(f"{x:.5f}")


def solve_gammas_first_success(
    gamma_list,
    m,
    N,
    static_threshold,
    static_horizon,
    static_criteria,
    fluc_threshold,
    fluc_horizon,
    fluc_criteria,
    record_washout,
    impact_threshold,
    sampling_rate,
    dt,
    test_impact_threshold,
    esn,
    model,
    all_train_data,
    washout=100,
    base_seed=42,
    n_jobs=None,
    start_inflight_per_gamma=1,
    max_inflight_cap=8,
    shuffle_seed=None,
    print_results=True,
    mp_context=None,
):
    """For each gamma, launch up to m randomized sims; keep the first success."""
    if mp_context is None:
        import multiprocessing as mp
        mp_context = mp.get_context("spawn")

    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 4) - 1)

    # shuffle gamma order
    shuffled = list(gamma_list)
    import random as _random
    (_random.Random(shuffle_seed).shuffle(shuffled)
     if shuffle_seed is not None else _random.shuffle(shuffled))

    manager = mp_context.Manager()
    done_flags = manager.dict({key5(g): False for g in shuffled})

    results_by_gamma = {key5(g): None for g in shuffled}
    submitted_counts = {key5(g): 0 for g in shuffled}
    running_counts = {key5(g): 0 for g in shuffled}
    orig_gamma = {key5(g): g for g in shuffled}
    keys = list(orig_gamma.keys())

    futures_by_key = {k: set() for k in keys}

    def remaining_keys():
        return [k for k in keys if (not done_flags[k]) and (submitted_counts[k] < m)]

    def current_per_key_cap():
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
        seed = base_seed + submitted_counts[k]
        fut = ex.submit(
            run_one_gamma_with_flag,
            submitted_counts[k] + 1,
            orig_gamma[k],
            N,
            static_threshold,
            static_horizon,
            static_criteria,
            fluc_threshold,
            fluc_horizon,
            fluc_criteria,
            record_washout,
            impact_threshold,
            sampling_rate,
            dt,
            test_impact_threshold,
            esn,
            model,
            all_train_data,
            done_flags,
            washout,
            seed,
            print_results,
        )
        submitted_counts[k] += 1
        running_counts[k] += 1
        futures_by_key[k].add(fut)
        future_to_key[fut] = k

    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp_context) as ex:
        future_to_key = {}

        # seed tasks
        for g in shuffled:
            k = key5(g)
            n0 = min(start_inflight_per_gamma, m - submitted_counts[k])
            for _ in range(n0):
                submit_one(ex, k)

        def fill_idle():
            while len(future_to_key) < n_jobs:
                cands = [k for k in keys if can_submit(k)]
                if not cands:
                    break
                _random.shuffle(cands)
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

        while future_to_key:
            done, _ = wait(set(future_to_key.keys()), return_when=FIRST_COMPLETED)

            for fut in done:
                k_done = future_to_key.pop(fut)
                futures_by_key[k_done].discard(fut)
                running_counts[k_done] = max(0, running_counts[k_done] - 1)

                r = fut.result()
                if r["reached"] and results_by_gamma[k_done] is None:
                    results_by_gamma[k_done] = r
                    done_flags[k_done] = True

                    # cancel the rest of the same-gamma tasks
                    for other in list(futures_by_key[k_done]):
                        if not other.done():
                            other.cancel()
                        futures_by_key[k_done].discard(other)

                fill_idle()

        # backfill failed ones
        for k in list(results_by_gamma.keys()):
            if results_by_gamma[k] is None:
                results_by_gamma[k] = {
                    "gamma": orig_gamma[k],
                    "impact_intervals": [],
                    "test_pred": None,
                    "reached": False,
                    "skipped": False,
                }

    return results_by_gamma


# ------------------------------
# Main script
# ------------------------------
def main():

    np.random.seed(1234)

    # ----- config -----
    gamma_train_pick = 0.01551
    train_gammas = [gamma_train_pick, 0.01801, 0.01901]

    sampling_rate = 100
    dt = 1.0 / sampling_rate

    esn_in_size = expand_state(np.zeros(4), 0, 0, np.zeros(4), np.zeros(4)).shape[1]
    esn_r_size = 400
    rho = 0.9
    leaky = 0.1
    dens = 1.0
    esn_in_scale = 1.1

    washout = 100
    train_impact_threshold = 1 / sampling_rate
    test_impact_threshold = 1 / sampling_rate

    noise_std = 3e-4
    ridge_alpha = 1e-8

    # ----- train readout until reaching target error -----
    train_error = np.inf
    all_train_data, esn, model = None, None, None
    while train_error > 1.05 * noise_std:
        all_train_data, esn, model, train_error = train_esn_model(
            train_gammas,
            noise_std,
            esn_in_size,
            esn_r_size,
            rho,
            leaky,
            dens,
            esn_in_scale,
            ridge_alpha,
            dt,
            train_impact_threshold,
            sampling_rate,
            washout,
        )
    print(f"Training completed with error: {train_error}")

    # ----- search over prediction gammas -----
    gamma_predict = np.arange(0.01001, 0.01951, 0.0005).tolist()

    N = 50  # per-gamma random trials
    static_threshold = 2000
    static_horizon = 200
    static_criteria = 1e-2

    fluc_threshold = 3000
    fluc_horizon = 5
    fluc_criteria = 1e-1

    record_washout = 50
    impact_threshold = record_washout + 50

    m = 50  # max attempts per gamma

    import multiprocessing as mp
    ctx = mp.get_context("spawn")

    results = solve_gammas_first_success(
        gamma_list=gamma_predict,
        m=m,
        N=N,
        static_threshold=static_threshold,
        static_horizon=static_horizon,
        static_criteria=static_criteria,
        fluc_threshold=fluc_threshold,
        fluc_horizon=fluc_horizon,
        fluc_criteria=fluc_criteria,
        record_washout=record_washout,
        impact_threshold=impact_threshold,
        sampling_rate=sampling_rate,
        dt=dt,
        test_impact_threshold=test_impact_threshold,
        esn=esn,
        model=model,
        all_train_data=all_train_data,
        washout=washout,
        base_seed=1234,
        n_jobs=max(1, (os.cpu_count() or 4) - 2),
        print_results=True,
        mp_context=ctx,
    )

    # ----- collect and plot bifurcation -----
    gamma_impact_intervals = {}
    for gamma in gamma_predict:
        r = results[float(f"{gamma:.5f}")]
        if r["reached"]:
            g = float(f"{r['gamma']:.5f}")
            gamma_impact_intervals.setdefault(g, []).extend(r["impact_intervals"])

    all_gammas, all_intervals = [], []
    for g, intervals in gamma_impact_intervals.items():
        all_gammas.extend([g] * len(intervals))
        all_intervals.extend(intervals)

    plot_bifurcation(np.array(all_gammas), np.array(all_intervals))


if __name__ == "__main__":
    # Force spawn (safe for macOS/Windows; on Linux this is also fine)
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # start method was already set elsewhere

    main()