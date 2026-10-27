# Primary tests (rl_mean − comparator, paired; Holm-corrected)

| scenario | comparator | metric | n | mean_diff | ci_lo | ci_hi | p | p_holm | significant |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stable | kademlia_epoch | J | 20 | 0.0097 | 0.0091 | 0.0102 | 1.91e-06 | 1.72e-05 | True |
| stable | tuned_heuristic | J | 20 | -0.0021 | -0.0030 | -0.0013 | 6.29e-05 | 0.000189 | True |
| stable | matched_random | J | 20 | 0.0149 | 0.0061 | 0.0295 | 9.54e-06 | 3.81e-05 | True |
| high_churn | kademlia_epoch | J | 20 | 0.0085 | 0.0065 | 0.0110 | 1.91e-06 | 1.72e-05 | True |
| high_churn | tuned_heuristic | J | 20 | -0.0030 | -0.0044 | -0.0017 | 0.000483 | 0.000483 | True |
| high_churn | matched_random | J | 20 | 0.0127 | 0.0100 | 0.0160 | 1.91e-06 | 1.72e-05 | True |
| heldout_medium | kademlia_epoch | J | 20 | 0.0089 | 0.0081 | 0.0098 | 1.91e-06 | 1.72e-05 | True |
| heldout_medium | tuned_heuristic | J | 20 | -0.0020 | -0.0027 | -0.0013 | 6.29e-05 | 0.000189 | True |
| heldout_medium | matched_random | J | 20 | 0.0126 | 0.0053 | 0.0235 | 3.81e-06 | 1.91e-05 | True |
