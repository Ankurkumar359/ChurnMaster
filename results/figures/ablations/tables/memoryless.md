# Memoryless held-out scenario (Task 12.3, exploratory)

`heldout_memoryless` = `heldout_medium` with Weibull shape 1.0 (frozen: 0.6); mean session lengths unchanged. TEST split, run 0, episodes 0..19. RL = rl_mean over the 3 canonical runs. 95% paired bootstrap CIs; p-values uncorrected (no Holm).

## Per-policy means and shape effect

| policy | J (heldout_memoryless) | J (heldout_medium) | ΔJ shape [95% CI] | p95 ms (heldout_memoryless) | p95 ms (heldout_medium) | Δp95 shape [95% CI] | success (heldout_memoryless) | pings/h (heldout_memoryless) |
|---|---|---|---|---|---|---|---|---|
| rl_mean | 0.9700 | 0.9710 | -0.0009 [-0.0018, -0.0000] | 557.3 | 549.3 | +8.0 [-15.1, +31.5] | 0.9997 | 27.9 |
| tuned_heuristic | 0.9725 | 0.9741 | -0.0017 [-0.0024, -0.0008] | 471.0 | 444.6 | +26.4 [-26.8, +84.2] | 0.9999 | 27.5 |
| kademlia_epoch | 0.9619 | 0.9616 | +0.0003 [-0.0008, +0.0014] | 480.3 | 459.8 | +20.4 [-29.8, +74.6] | 0.9997 | 380.9 |
| matched_random | 0.9544 | 0.9610 | -0.0065 [-0.0169, +0.0010] | 737.4 | 676.2 | +61.2 [-17.7, +145.4] | 0.9964 | 28.3 |
| never_replace | 0.9438 | 0.9463 | -0.0025 [-0.0250, +0.0165] | 780.6 | 758.7 | +21.9 [-92.8, +133.6] | 0.9912 | 0.0 |
| uniform_random | 0.9556 | 0.9566 | -0.0010 [-0.0017, -0.0003] | 552.7 | 522.0 | +30.7 [-4.5, +64.0] | 0.9999 | 436.5 |
| oracle_heuristic | 0.9724 | 0.9733 | -0.0009 [-0.0019, +0.0002] | 428.3 | 394.7 | +33.6 [-9.2, +78.9] | 0.9999 | 23.4 |

## RL (rl_mean) vs references

| reference | scenario | ΔJ [95% CI] | p (J) | Δp95 ms [95% CI] | Δpings/h [95% CI] |
|---|---|---|---|---|---|
| tuned_heuristic | heldout_memoryless | -0.0024 [-0.0034, -0.0015] | 9.54e-06 | +86.2 [+38.5, +137.9] | +0.4 [-3.3, +4.0] |
| tuned_heuristic | heldout_medium | -0.0031 [-0.0043, -0.0021] | 4.77e-05 | +104.6 [+55.4, +153.0] | +0.6 [-2.3, +3.5] |
| kademlia_epoch | heldout_memoryless | +0.0082 [+0.0074, +0.0088] | 1.91e-06 | +77.0 [+36.6, +122.5] | -352.9 [-366.1, -338.5] |
| kademlia_epoch | heldout_medium | +0.0094 [+0.0086, +0.0102] | 1.91e-06 | +89.4 [+50.5, +131.6] | -369.0 [-383.2, -355.5] |
| matched_random | heldout_memoryless | +0.0156 [+0.0090, +0.0255] | 1.91e-06 | -180.2 [-249.5, -121.2] | -0.4 [-3.0, +2.4] |
| matched_random | heldout_medium | +0.0100 [+0.0075, +0.0128] | 1.91e-06 | -127.0 [-175.9, -91.6] | -4.5 [-7.7, -1.5] |
| never_replace | heldout_memoryless | +0.0262 [+0.0138, +0.0455] | 1.91e-06 | -223.3 [-297.0, -156.6] | +27.9 [+25.1, +30.7] |
| never_replace | heldout_medium | +0.0246 [+0.0154, +0.0367] | 1.91e-06 | -209.4 [-294.0, -140.9] | +20.0 [+16.8, +23.8] |
| uniform_random | heldout_memoryless | +0.0144 [+0.0135, +0.0154] | 1.91e-06 | +4.6 [-25.1, +35.9] | -408.6 [-422.2, -395.7] |
| uniform_random | heldout_medium | +0.0143 [+0.0131, +0.0155] | 1.91e-06 | +27.3 [-2.8, +56.4] | -415.3 [-431.6, -399.2] |
| oracle_heuristic | heldout_memoryless | -0.0024 [-0.0029, -0.0019] | 1.91e-06 | +129.0 [+91.6, +168.9] | +4.5 [+1.5, +7.3] |
| oracle_heuristic | heldout_medium | -0.0023 [-0.0029, -0.0018] | 9.54e-06 | +154.6 [+123.9, +184.1] | +2.1 [-0.2, +4.5] |

## Gap shift: [RL − ref]_heldout_memoryless − [RL − ref]_heldout_medium

| reference | ΔΔJ [95% CI] | p (J) | ΔΔp95 ms [95% CI] |
|---|---|---|---|
| tuned_heuristic | +0.0007 [-0.0004, +0.0018] | 0.261 | -18.4 [-77.4, +35.4] |
| kademlia_epoch | -0.0012 [-0.0021, -0.0003] | 0.0172 | -12.4 [-58.8, +32.9] |
| matched_random | +0.0056 [-0.0021, +0.0163] | 0.571 | -53.2 [-135.7, +21.6] |

Caveat: One held-out scenario, 20 TEST seeds, no multiple-comparison correction; the canonical policies were neither retrained nor re-tuned on it.
