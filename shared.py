
import itertools
import math
from functools import lru_cache


def round_robin_groups(r, a):
    """Group sizes when a live workers are spread over min(r, a) partitions by round robin."""
    x = min(r, a)
    return [a // x + (1 if i < a % x else 0) for i in range(x)]

# print(round_robin_groups(3, 10))
# [4, 3, 3], 4 workers to first parition, etc.

# print(round_robin_groups(5, 2))
# [1, 1], 1 worker to first parition, etc


def failure_sets(a, P):
    """Every possible set of failed workers (out of a live workers) with its probability.
    Workers fail independently; if all a would fail, one (chosen at random) survives."""
    outcomes = {}
    for failed in itertools.product([False, True], repeat=a):
        f = sum(failed)
        p = P ** f * (1 - P) ** (a - f)
        if f == a:                                  # everyone would fail: one survives
            for survivor in range(a):
                s = tuple(i != survivor for i in range(a))
                outcomes[s] = outcomes.get(s, 0) + p / a
        else:
            outcomes[failed] = outcomes.get(failed, 0) + p
    return outcomes.items()

#print(failure_sets(3, 0.2))
# F/F/F --> 0.512 = 0.8^3, etc
# cannot return 
# dict_items([((False, False, False), 0.5120000000000001), ((False, False, True), 0.12800000000000003), ((False, True, False), 0.12800000000000003), ((False, True, True), 0.03466666666666667), ((True, False, False), 0.12800000000000003), ((True, False, True), 0.03466666666666667), ((True, True, False), 0.03466666666666667)])










def split_into_blocks(N, m):
    """Split partitions 1..N into m consecutive blocks (sizes differ by at most 1).

    >>> split_into_blocks(10, 3)
    [[1, 2, 3, 4], [5, 6, 7], [8, 9, 10]]
    """
    sizes = [N // m + (1 if i < N % m else 0) for i in range(m)]
    blocks, start = [], 1
    for size in sizes:
        blocks.append(list(range(start, start + size)))
        start += size
    return blocks


def own_block_order(block, left):
    """Two-pointer order on the pair's own block: left worker goes up, right goes down.

    >>> own_block_order([1, 2, 3, 4], left=True), own_block_order([1, 2, 3, 4], left=False)
    ([1, 2, 3, 4], [4, 3, 2, 1])
    """
    return block if left else block[::-1]


def visiting_order(block, left):
    """Order for visiting another pair's block: start in the middle, go outward.

    >>> visiting_order([1, 2, 3, 4], left=True), visiting_order([1, 2, 3, 4], left=False)
    ([2, 1, 4, 3], [3, 4, 1, 2])
    """
    middle = (len(block) - 1) // 2
    lower_half = block[:middle + 1]
    upper_half = block[middle + 1:]
    if left:
        return lower_half[::-1] + upper_half[::-1]
    return upper_half + lower_half


def pair_schedules(N, k):
    """Task lists for k workers (k even) and N partitions; list i is worker i+1's order.

    >>> pair_schedules(8, 4)
    [[1, 2, 3, 4, 6, 5, 8, 7], [4, 3, 2, 1, 7, 8, 5, 6], [5, 6, 7, 8, 2, 1, 4, 3], [8, 7, 6, 5, 3, 4, 1, 2]]
    """
    if k % 2 != 0:
        raise ValueError("k must be even")
    m = k // 2                                   # number of pairs = number of blocks
    if N < m:
        raise ValueError("need at least k/2 partitions, so every block is non-empty")
    blocks = split_into_blocks(N, m)

    schedules = []
    for pair in range(m):
        left_schedule, right_schedule = [], []
        for stage in range(m):
            block = blocks[(pair + stage) % m]   # own block first, then rotate
            if stage == 0:
                left_schedule += own_block_order(block, left=True)
                right_schedule += own_block_order(block, left=False)
            else:
                left_schedule += visiting_order(block, left=True)
                right_schedule += visiting_order(block, left=False)
        schedules += [left_schedule, right_schedule]
    return schedules






# - need this?

# this is the recursive form for the algorithm
def B_expected_rounds_recursive(schedules, N, P):
    """Exact expected rounds by recursion over (done partitions, live workers). Small N only.

    >>> round(B_expected_rounds_recursive(pair_schedules(8, 4), 8, 0.1), 4)
    2.8283
    """
    from functools import lru_cache
    k = len(schedules)
    all_done = (1 << N) - 1                      # bit p-1 set = partition p is done

    @lru_cache(maxsize=None)
    def E(done, alive):
        if done == all_done:
            return 0.0
        live = [w for w in range(k) if alive >> w & 1]
        # each live worker takes the first partition on its list that isn't done
        target = [next(p for p in schedules[w] if not done >> (p - 1) & 1) for w in live]
        total = 1.0                              # this round
        for failed, p in failure_sets(len(live), P):
            new_done, new_alive = done, alive
            for i, w in enumerate(live):
                if failed[i]:
                    new_alive &= ~(1 << w)       # worker w is gone
                else:
                    new_done |= 1 << (target[i] - 1)
            total += p * E(new_done, new_alive)
        return total

    return E(0, (1 << k) - 1)


def targets_this_round(schedules, done, live):
    """The partition each live worker takes: the first one on its list that isn't done.

    >>> targets_this_round([[1, 2, 3], [3, 2, 1]], done=0b001, live=[0, 1])   # partition 1 done
    [2, 3]
    """
    return [next(p for p in schedules[w] if not done >> (p - 1) & 1) for w in live]


def B_worst_rounds(schedules, N, f):
    """Most rounds B can take with exactly f failures, over every 'which' and 'when'.

    >>> B_worst_rounds(pair_schedules(5, 2), 5, 1)       # same as opposite ends: fail in round 1
    5
    >>> B_worst_rounds(pair_schedules(12, 6), 12, 4)     # e.g. w1, w2, w3, w5 fail in round 1
    8
    """
    k = len(schedules)
    all_done = (1 << N) - 1

    @lru_cache(maxsize=None)
    def W(done, alive, f_left):
        if done == all_done:
            return 0
        live = [w for w in range(k) if alive >> w & 1]
        targets = targets_this_round(schedules, done, live)
        worst_next = 0
        for x in range(min(f_left, len(live) - 1) + 1):          # how many fail this round
            for failed in itertools.combinations(range(len(live)), x):   # which ones
                new_done, new_alive = done, alive
                for i, w in enumerate(live):
                    if i in failed:
                        new_alive &= ~(1 << w)                    # worker w is gone
                    else:
                        new_done |= 1 << (targets[i] - 1)         # its partition is finished
                worst_next = max(worst_next, W(new_done, new_alive, f_left - x))
        return 1 + worst_next

    return W(0, (1 << k) - 1, f)


# ---------- Completion-time distribution (exact) ----------

def A_time_distribution(N, k, P):
    """Method A: {t: P(T = t)}, the chance the job takes exactly t rounds.

    >>> d = A_time_distribution(5, 2, 0.1)
    >>> [(t, round(p, 4)) for t, p in sorted(d.items())]
    [(3, 0.6561), (4, 0.1539), (5, 0.19)]
    """
    @lru_cache(maxsize=None)
    def D(r, a):
        if r == 0:
            return {0: 1.0}
        if a == 1:
            return {r: 1.0}                              # last worker can't fail
        groups = round_robin_groups(r, a)
        partition_of_worker = [g for g, size in enumerate(groups) for _ in range(size)]
        out = {}
        for failed, p in failure_sets(a, P):             # every outcome of this round
            finished = len({partition_of_worker[i] for i in range(a) if not failed[i]})
            for t, q in D(r - finished, a - sum(failed)).items():
                out[t + 1] = out.get(t + 1, 0) + p * q   # this round adds 1
        return out
    return D(N, k)


def B_time_distribution(schedules, N, P):
    """Method B: {t: P(T = t)}, the chance the job takes exactly t rounds. Small N only.

    >>> d = B_time_distribution(pair_schedules(5, 2), 5, 0.1)
    >>> [(t, round(p, 4)) for t, p in sorted(d.items())]
    [(3, 0.6561), (4, 0.1539), (5, 0.19)]
    """
    k = len(schedules)
    all_done = (1 << N) - 1

    @lru_cache(maxsize=None)
    def D(done, alive):
        if done == all_done:
            return {0: 1.0}
        live = [w for w in range(k) if alive >> w & 1]
        targets = targets_this_round(schedules, done, live)
        out = {}
        for failed, p in failure_sets(len(live), P):
            new_done, new_alive = done, alive
            for i, w in enumerate(live):
                if failed[i]:
                    new_alive &= ~(1 << w)
                else:
                    new_done |= 1 << (targets[i] - 1)
            for t, q in D(new_done, new_alive).items():
                out[t + 1] = out.get(t + 1, 0) + p * q
        return out
    return D(0, (1 << k) - 1)


def percentile_rounds(distribution, level):
    """Smallest t with P(T <= t) >= level, e.g. level=0.95 for "95% of runs finish within t".

    >>> percentile_rounds({3: 0.6561, 4: 0.1539, 5: 0.19}, 0.8)
    4
    """
    cumulative = 0.0
    for t in sorted(distribution):
        cumulative += distribution[t]
        if cumulative >= level - 1e-12:
            return t
    return max(distribution)


# ---------- Timed worst case: chance decides HOW MANY fail each round, worst case decides WHICH ----------

def prob_x_fail(a, P, x):
    """Chance that exactly x of a live (working) workers fail this round; the last one never fails.

    >>> round(prob_x_fail(2, 0.1, 1), 4)       # 2 * 0.1 * 0.9 + 0.1**2
    0.19
    """
    p = math.comb(a, x) * P ** x * (1 - P) ** (a - x)
    if x == a - 1:
        p += P ** a                                     # 'all fail' becomes 'all but one fail'
    return p


def A_timed_worst(N, k, P):
    """Method A: expected rounds when the worst workers fail, at the real failure rate.

    >>> round(A_timed_worst(5, 2, 0.1), 4)      # 2 workers: same as the expected time
    3.5339
    """
    @lru_cache(maxsize=None)
    def V(r, a):
        if r == 0:
            return 0.0
        if a == 1:
            return float(r)
        groups = round_robin_groups(r, a)
        partition_of_worker = [g for g, size in enumerate(groups) for _ in range(size)]
        total = 1.0
        for x in range(a):                              # how many fail (chance)
            worst = max(V(r - len({partition_of_worker[i] for i in range(a) if i not in failed}), a - x)
                        for failed in itertools.combinations(range(a), x))   # which ones (worst)
            total += prob_x_fail(a, P, x) * worst
        return total
    return V(N, k)


def B_timed_worst(schedules, N, P):
    """Method B: expected rounds when the worst workers fail, at the real failure rate.

    >>> round(B_timed_worst(pair_schedules(12, 6), 12, 0.3), 4)
    6.4874
    """
    k = len(schedules)
    all_done = (1 << N) - 1

    @lru_cache(maxsize=None)
    def V(done, alive):
        if done == all_done:
            return 0.0
        live = [w for w in range(k) if alive >> w & 1]
        targets = targets_this_round(schedules, done, live)
        total = 1.0
        for x in range(len(live)):
            worst = 0.0
            for failed in itertools.combinations(range(len(live)), x):
                new_done, new_alive = done, alive
                for i, w in enumerate(live):
                    if i in failed:
                        new_alive &= ~(1 << w)
                    else:
                        new_done |= 1 << (targets[i] - 1)
                worst = max(worst, V(new_done, new_alive))
            total += prob_x_fail(len(live), P, x) * worst
        return total
    return V(0, (1 << k) - 1)
