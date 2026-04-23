import numpy as np
from pykalman import KalmanFilter, UnscentedKalmanFilter


# ---------------- STANDARD KF ----------------
def run_kf_linear(y_sequence, config, use_em=False):
    A = config["dataset"]["A"]
    H = config["dataset"]["H"]
    Q = config["dataset"]["Q"]
    R = config["dataset"]["R"]

    kf = KalmanFilter(
        transition_matrices=[A],
        observation_matrices=[H],
        transition_covariance=[[Q]],
        observation_covariance=[[R]],
        initial_state_mean=0,
        initial_state_covariance=1
    )

    if use_em:
        kf = kf.em(y_sequence, n_iter=5)

    state_means, _ = kf.filter(y_sequence)

    return state_means.flatten()


# ---------------- UKF (OPTIONAL EXTENSION) ----------------
def run_ukf(y_sequence, transition_fn, observation_fn):
    ukf = UnscentedKalmanFilter(
        transition_functions=transition_fn,
        observation_functions=observation_fn
    )

    state_means, _ = ukf.filter(y_sequence)

    return state_means.flatten()