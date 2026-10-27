# Headroom gates (high_churn, split=calibration, n_seeds=10)

G1: PASS (route=J, best=oracle_heuristic)

| candidate | dJ vs never_replace [95% CI] | p95 never_replace − candidate (ms) | p95_rel | J route | p95 route | passes |
| --- | --- | --- | --- | --- | --- | --- |
| kademlia_epoch | +0.0274 [+0.0204, +0.0344] | +431.6 [+328.0, +517.7] | 0.4099 | True | True | True |
| always_replace | +0.0182 [+0.0105, +0.0256] | +419.8 [+311.0, +510.5] | 0.3957 | True | True | True |
| oracle_heuristic | +0.0424 [+0.0342, +0.0502] | +581.0 [+486.7, +673.7] | 0.5720 | True | True | True |


G2: PASS (route=J, limited_headroom=False)

| scenario | kind | dJ oracle − kademlia_epoch [95% CI] | ping ratio | passed |
| --- | --- | --- | --- | --- |
| high_churn | gate | +0.0150 [+0.0119, +0.0188] | 0.0939 | True |
| stable | info | +0.0106 [+0.0092, +0.0117] | 0.0215 | True |


Thresholds: g1_min_dJ=0.01, g1_min_p95_rel=0.2, g2_min_dJ=0.01, g2_max_ping_ratio=0.6, g2_J_tol=0.005
