from dataclasses import dataclass

@dataclass
class EnvConfig:
    # ---- overlay / topology ----
    n_nodes: int = 500
    k: int = 16
    m: int = 4            # Reduced candidate candidates pool from 8 -> 4 (shrinks action space from 129 to 65)
    episode_len: int = 300
    warmup: int = 300

    # ---- workload ----
    lookups_per_step: int = 5
    probes_per_step: int = 2     # Increased from 1 -> 2 (refreshes routing table health twice as fast)
    max_tries: int = 2           # Increased from 1 -> 2 (allows fallback routing before declaring lookup failure)
    drop_prob: float = 0.02

    # ---- churn model (Weibull sessions, heavy tailed) ----
    churn_scale: float = 2.0
    stable_frac: float = 0.3
    stable_session: float = 1800.0
    flaky_session: float = 180.0
    weibull_shape: float = 0.6
    down_mean: float = 90.0

    # ---- latency model (ms) ----
    base_rtt_ms: float = 20.0
    rtt_per_unit_ms: float = 200.0
    rtt_noise_ms: float = 10.0
    timeout_ms: float = 500.0
    tail_hops: int = 2
    tail_hop_ms: float = 120.0
    lat_norm_ms: float = 1000.0

    # ---- reward weights ----
    alpha: float = 1.0           # Lookup success
    beta: float = 0.4            # Increased from 0.3 (higher sensitivity to latency/timeouts)
    gamma: float = 0.30          # Increased from 0.05 (suppresses hyperactive churn)
    delta: float = 0.50          # Increased from 0.20 (strongly penalizes holding dead peers)

    # ---- observation normalisers ----
    age_scale: float = 3600.0
    fail_norm: float = 5.0
    since_norm: float = 60.0
    ablate_latency: bool = False