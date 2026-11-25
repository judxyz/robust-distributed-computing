# How the simulations and graphs work

This document describes how each notebook computes its results and produces its graphs.
All shared functions live in `shared.py`; graphs are saved to `images/`.

## Model (used everywhere)

- **N** partitions, **k** worker nodes, **P** = failure probability. Each partition takes time **T**; work proceeds in synchronized rounds, and each working worker processes one partition per round. All times are in rounds (multiply by T).
- Each **working** worker fails with probability P in a round, independently. A failed worker is gone for good, and the partition it was working on is not finished that round. Idle workers cannot fail.
- **The last live worker never fails**, so the job always completes. In code:
  - exact calculations (`failure_sets`, `prob_x_fail`) turn "all fail" into "all but one fail";
  - simulations skip the failure check when only one worker is left.
- Tasks are idempotent: a partition can be processed by two workers at once (a backup), and any worker can redo a failed worker's partition.

## The two methods

**Method A: lower bound (benchmark).** A central scheduler with full monitoring. Each round, every live worker works on a different unfinished partition. When fewer partitions are left than live workers, all live workers are spread as evenly as possible over the remaining partitions (round robin, `round_robin_groups`); the extras act as backups. Because partitions are identical, the state is just `(r, a)` = (partitions left, live workers). This is the best possible scheduler for the **expected** completion time (checked by exhaustive search for up to 4 workers).

**Method B: 2 pointer (pair schedules).** Each worker gets a fixed list of partitions in advance (`pair_schedules`); the only monitoring is a done flag per partition.

1. Split the k workers into k/2 pairs and the partitions into k/2 consecutive blocks.
2. Each pair starts on its own block with two pointers: the left worker goes up from the block's start, the right worker down from its end.
3. Each pair then visits the other blocks in rotating order (pair i visits blocks i+1, i+2, …).
4. In a visited block, the pair starts in the middle and works outward (block `[1, 2, 3, 4]`: left worker `2, 1, 4, 3`, right worker `3, 4, 1, 2`).
5. At run time, each live worker takes the first partition on its list that is not done (`targets_this_round`).

Every list contains every partition, so any single survivor finishes the job. The state is `(done, alive)` = which partitions are done and which workers are alive, both stored as bitmasks. With k = 2 this is the original opposite-ends algorithm.

## Two kinds of calculation

- **Exact recursion.** Starting from a state, list every possible outcome of the current round with its probability, and combine the values of the states they lead to:

  `value(state) = 1 + Σ over outcomes [ probability × value(next state) ]`

  The `1` is the current round. Results are cached with `lru_cache`, so each state is computed once. Practical up to about 10 workers for method A and about 16 partitions / 8 workers for method B.
- **Monte Carlo simulation.** Play out the schedule many times with random failures and average the number of rounds; the ± value is the 95% confidence half-width, `1.96 × std / √runs`. Used to check the exact values and for sizes too large for the recursion.

---

## `a_2worker_example.ipynb`: two workers, three partitions

**Graph `1_expected_time.png`.** Evaluates the 3-partition expected-time formula for 1,000 values of P from 0 to 1 (no simulation). It shows the corrected formula, `P×3T + (1−P)(P×3T + (1−P)×2T) = (3 − (1−P)²)T`, and, dashed, the original formula, which ordered the failure branches incorrectly.

**Graph `2_monte_carlo.png`.** A Monte Carlo simulation of the schedule itself:

- `simulate_run` keeps two pointers, `left` (W1) and `right` (W2), which move inward each round unless that worker has failed. Failures are checked only while both workers are alive (the last-survivor rule).
- The loop runs until the pointers cross; the number of rounds is the completion time. No finish times are written into the code, so the simulation independently checks the formula.
- It runs 10,000 times at each of 11 fixed values of P (0, 0.1, …, 1). P is a fixed grid, not random: the randomness is in the failures. The averages (points) land on the corrected formula (line).

**`general_2w`.** The recursive form of the 2-worker expected time, `E(n) = (1 − q)·n + q·E(n − 1)` with `q = (1 − P)²` and `E(⌈N/2⌉) = ⌈N/2⌉`, where `n` is the total number of rounds if the first failure happens in the current round. For 5 partitions at P = 0.1 it gives 3.5339, equal to the closed form `N − q − q² − … − q^⌊N/2⌋`.

## `b_average_times.ipynb`: expected (average) completion time

**Method A, exact (`A_expected_rounds_recursive`).** Recursion over `(r, a)`:

- `E(0, a) = 0`; `E(r, 1) = r` (the last worker cannot fail).
- Otherwise assign workers by the round-robin rule, then go through every set of failed workers from `failure_sets(a, P)`. A partition is finished if at least one of its workers survives. `E(r, a) = 1 + Σ probability × E(r − finished, a − failed)`.

**Method A, simulated (`A_simulate_run`).** Plays out the same rule with random failures; `A_expected_rounds_simulated` averages 100,000 runs.

**Method B, exact (`B_expected_rounds_recursive`).** Recursion over `(done, alive)`: each live worker takes the first undone partition on its list; for every set of failed workers, finished partitions are added to `done` (a union) and failed workers removed from `alive`. `E(done, alive) = 1 + Σ probability × E(done ∪ finished, alive − failed)`.

**Method B, simulated (`B_simulate_run`).** Plays out the schedules with random failures, checking workers in a random order for the last-survivor rule; `B_expected_rounds_simulated` averages 40,000 runs.

**Tables.** Exact against simulated values for each method (they agree within the ± interval), and B against A with the gap `100 × (B / A − 1)` %.

**Graphs** (P = 0.2):

- `b_average_increasing_partitions.png`: k = 6 workers, N = 6 to 36 partitions.
- `b_average_increasing_workers.png`: N = 20 partitions, k = 2 to 10 workers (even only, as pair schedules need pairs).

Method A uses the exact recursion (simulation above 10 workers); method B is simulated with 20,000 runs per point and drawn with 95% error bars.

## `c_worst_case_times.ipynb`: expected time when the worst workers fail (timed worst case)

This measure keeps the real timing of failures but makes the unluckiest choice of which workers fail:

- **chance** decides **how many** workers fail in each round, with the real probabilities: `prob_x_fail(a, P, x)` = binomial chance that x of a working workers fail, with "all fail" counted as "all but one fail";
- **the worst case** decides **which** workers those are, choosing the set that makes the rest of the job slowest.

`V(state) = 1 + Σ over x of prob_x_fail(a, P, x) × max over which x workers fail of V(next state)`

- `A_timed_worst(N, k, P)`: recursion over `(r, a)`; the choice of which workers matters only at the end, when some partitions have backups (failing a lone worker loses its partition; failing a backup does not).
- `B_timed_worst(schedules, N, P)`: recursion over `(done, alive)`; which workers fail matters more, because losing certain workers leaves visiting pairs with identical orders that duplicate each other's work.

Properties that check the code: for 2 workers it equals the expected time exactly (which of two symmetric workers fails makes no difference); it is always at least the expected time.

**Tables and graphs** (P = 0.2), exact, so there are no error bars:

- `c_worst_increasing_partitions.png`: k = 6 workers, N = 6 to 36 partitions.
- `c_worst_increasing_workers.png`: N = 20 partitions, k = 2 to 10 workers.

Notes for interpretation:

- This is a stress test, not typical behaviour: real failures hit random workers.
- Under this measure method A is a benchmark rather than a strict lower bound: a scheduler that keeps spare workers idle near the end (instead of using them as backups) does 0–1% better, because idle workers cannot fail and the worst case never fails a backup whose partner survives.

## `d_charts_worstcase.ipynb`: method B by total number of failures

Two bar charts for method B (N = 12, k = 6, P = 0.1), with the **total number of failures f over the whole run** (0 to k − 1) on the x-axis. Each bar is labelled with P(F = f), the chance of exactly f failures.

- **`d_average_time_by_failures.png`**: E[T | F = f], the average completion time of runs with exactly f failures. `B_time_by_failure_count` runs the method-B recursion but carries, for each f, both the probability and the probability-weighted time; dividing the two gives the conditional average.
- **`d_worst_time_by_failures.png`**: W_B(f) from `B_worst_rounds`, the most rounds possible with exactly f failures, found by exhaustive search over how many workers fail in each round and which ones.

Weighting chart 1's bars by P(F = f) gives the overall expected time. Note: f counts every failed worker, including a backup whose partition another worker finishes in the same round (a failure that costs no time). Such runs are charged the worst case for their f, so the per-f worst cases are pessimistic.

## `e_completion_time_distribution.ipynb`: completion-time distribution

The exact probability of each completion time for both methods (N = 12, k = 6, P = 0.2), with everything random: how many workers fail, which ones, and when.

- `A_time_distribution(N, k, P)` and `B_time_distribution(schedules, N, P)` use the same recursions as the expected time, but each state returns a table `{t: P(T = t)}` instead of an average: every outcome's table is shifted by one round and weighted by its probability.
- **Table 1:** P(T = t) and P(T ≤ t) for each t. The expected time read off the distribution matches the exact recursion (4.0837 for A, 4.3070 for B).
- **Table 2:** probabilistic upper bounds from `percentile_rounds`: the smallest t with P(T ≤ t) ≥ level, e.g. "95% of runs finish within 8 rounds". The 100% row is the hard upper bound.
- **Graph `e_completion_time_distribution.png`:** step chart of P(T ≤ t) against t for both methods. Method B loses mostly in the middle (3–6 rounds), where duplicated visits cost time; from 8 rounds on the two curves coincide, because those runs have lost almost all their workers.

Considerations: exact only for small cases (simulate otherwise, with confidence bands); rounds are whole numbers, so percentiles move in steps and can tie even when probabilities differ; the tail is shaped by the model assumptions, especially the last-survivor rule, which caps the time at N.