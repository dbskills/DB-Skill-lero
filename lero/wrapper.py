#!/usr/bin/env python3
"""Lero Query Optimizer Skill Wrapper.

Generates candidate plans via pg_hint_plan Leading hints with connected join
orders. In training mode, executes plans with EXPLAIN ANALYZE, collects actual
latencies, and fine-tunes the Lero model online. State is persisted across
invocations (model weights, replay buffer).

Concurrency: persistent state (model weights + replay buffer) is guarded by a
file lock so that parallel wrapper invocations cannot corrupt each other's
writes or read half-written files. Scoring loads the model under a *shared*
lock (parallel reads); the training critical section runs under an *exclusive*
lock (serialized mutation). See acquire_lock()/release_lock().
"""

import argparse
import fcntl
import json
import os
import pickle
import random
import sys
import time
import warnings
from contextlib import redirect_stdout
from io import StringIO
from itertools import permutations

warnings.filterwarnings("ignore")

import psycopg2
import sqlglot
from sqlglot import exp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import LeroModel


def parse_args():
    parser = argparse.ArgumentParser(description="Lero Query Optimizer")
    parser.add_argument("--dsn", required=True, help="Database connection string")
    parser.add_argument("--query", required=True, help="SQL query to optimize")
    parser.add_argument("--config", required=False, help="Model directory path")
    parser.add_argument("--optimize-only", action="store_true",
                        help="Inference-only: skip query execution and training")
    return parser.parse_args()


def get_model_dir(args):
    if args.config:
        return args.config
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "reproduce", "imdb_pw")


def explain_plan(dsn, query, hint_str=""):
    conn = psycopg2.connect(dsn)
    conn.set_client_encoding('UTF8')
    try:
        cur = conn.cursor()
        sql = query.strip().rstrip(';').strip()
        if hint_str:
            sql = f"/*+ {hint_str} */ EXPLAIN (COSTS TRUE, FORMAT JSON) {sql}"
        else:
            sql = f"EXPLAIN (COSTS TRUE, FORMAT JSON) {sql}"
        cur.execute(sql)
        result = cur.fetchone()[0]
        if isinstance(result, list) and len(result) == 2:
            result = [result[1]]
        return json.dumps(result)
    finally:
        conn.close()


def explain_analyze(dsn, query, hint_str=""):
    conn = psycopg2.connect(dsn)
    conn.set_client_encoding('UTF8')
    try:
        cur = conn.cursor()
        cur.execute("SET statement_timeout TO 300000")
        sql = query.strip().rstrip(';').strip()
        if hint_str:
            sql = f"/*+ {hint_str} */ EXPLAIN (ANALYZE, TIMING, VERBOSE, COSTS, SUMMARY, FORMAT JSON) {sql}"
        else:
            sql = f"EXPLAIN (ANALYZE, TIMING, VERBOSE, COSTS, SUMMARY, FORMAT JSON) {sql}"
        cur.execute(sql)
        result = cur.fetchone()[0]
        if isinstance(result, list) and len(result) == 2:
            result = [result[1]]
        return json.dumps(result), result[0]["Execution Time"]
    finally:
        conn.close()


def extract_join_graph(query):
    """Extract table aliases and join edges using sqlglot."""
    tree = sqlglot.parse_one(query)
    if not isinstance(tree, exp.Select):
        return [], []

    aliases = []
    alias_set = set()

    def _collect_tables(src):
        if isinstance(src, exp.Table):
            a = src.alias_or_name.lower()
            if a and a not in alias_set:
                aliases.append(a)
                alias_set.add(a)
        elif isinstance(src, exp.Subquery):
            if isinstance(src.this, exp.Select):
                a = src.alias_or_name.lower()
                if a and a not in alias_set:
                    aliases.append(a)
                    alias_set.add(a)
            elif isinstance(src.this, exp.Table):
                a = src.alias_or_name.lower()
                if a:
                    if a not in alias_set:
                        aliases.append(a)
                        alias_set.add(a)
                else:
                    _collect_tables(src.this)

    def _walk_joins(join_list):
        for join in join_list:
            src = join.this
            _collect_tables(src)
            if isinstance(src, exp.Table):
                _walk_joins(src.args.get('joins') or [])
            elif isinstance(src, exp.Subquery) and isinstance(src.this, exp.Table):
                if not src.alias_or_name:
                    _walk_joins(src.this.args.get('joins') or [])

    from_clause = tree.args.get('from_')
    if from_clause:
        _collect_tables(from_clause.this)
        src = from_clause.this
        if isinstance(src, exp.Table):
            _walk_joins(src.args.get('joins') or [])
        elif isinstance(src, exp.Subquery) and isinstance(src.this, exp.Table):
            if not src.alias_or_name:
                _walk_joins(src.this.args.get('joins') or [])

    _walk_joins(tree.args.get('joins') or [])

    edges = set()

    def add_edge(col_a, col_b):
        if col_a.table and col_b.table:
            a1, a2 = col_a.table.lower(), col_b.table.lower()
            if a1 in alias_set and a2 in alias_set and a1 != a2:
                edges.add(tuple(sorted([aliases.index(a1), aliases.index(a2)])))

    def _collect_join_edges(join_node):
        on = join_node.args.get('on')
        if on:
            for node in on.walk():
                if isinstance(node, exp.EQ):
                    left, right = node.left, node.right
                    if isinstance(left, exp.Column) and isinstance(right, exp.Column):
                        add_edge(left, right)
        src = join_node.this
        if isinstance(src, exp.Subquery) and isinstance(src.this, exp.Table):
            for j in (src.this.args.get('joins') or []):
                _collect_join_edges(j)
        elif isinstance(src, exp.Table):
            for j in (src.args.get('joins') or []):
                _collect_join_edges(j)

    where = tree.args.get('where')
    if where:
        for node in where.walk():
            if isinstance(node, exp.EQ):
                left, right = node.left, node.right
                if isinstance(left, exp.Column) and isinstance(right, exp.Column):
                    add_edge(left, right)

    joins = tree.args.get('joins') or []
    for join in joins:
        _collect_join_edges(join)

    return aliases, list(edges)


def is_connected_join_order(order, edges):
    if len(order) <= 1:
        return True
    joined = {order[0]}
    for t in order[1:]:
        if not any(tuple(sorted([t, j])) in edges for j in joined):
            return False
        joined.add(t)
    return True


def generate_connected_join_orders(aliases, edges, candidate_limit=100):
    n = len(aliases)
    if n <= 1:
        return [tuple(aliases)]
    if n <= 6:
        all_orders = []
        for perm in permutations(range(n)):
            if is_connected_join_order(perm, edges):
                all_orders.append(tuple(aliases[i] for i in perm))
        return all_orders

    # For n > 6: constructive random sampling from the frontier.
    # Each step picks the next table from the set of tables adjacent to the
    # already-joined set, guaranteeing connectedness on every attempt.
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    orders = []
    seen = set()
    original = tuple(range(n))
    if is_connected_join_order(original, edges):
        orders.append(tuple(aliases[i] for i in original))
        seen.add(original)

    # If the original order isn't connected, the graph is disconnected — no
    # connected orders exist, so return early without wasting attempts.
    if not orders and not any(adj[i] for i in range(n)):
        return [tuple(aliases)]

    attempts = 0
    while len(orders) < candidate_limit and attempts < candidate_limit * 10:
        start = random.randrange(n)
        order = [start]
        joined = {start}
        frontier = set(adj[start])

        while len(order) < n:
            if not frontier:
                break
            nxt = random.choice(list(frontier))
            frontier.discard(nxt)
            order.append(nxt)
            joined.add(nxt)
            frontier.update(adj[nxt] - joined)

        if len(order) == n:
            perm_t = tuple(order)
            if perm_t not in seen:
                seen.add(perm_t)
                orders.append(tuple(aliases[i] for i in perm_t))
        attempts += 1

    if not orders:
        orders = [tuple(aliases)]

    return orders


REPLAY_CAP = 1000
TRAIN_TRIGGER = 100

# Name of the lock file used to coordinate parallel invocations. A process
# holds a shared (LOCK_SH) lock while reading model files for scoring, and an
# exclusive (LOCK_EX) lock while mutating persistent state (replay buffer +
# model weights). fcntl.flock locks are per-fd and auto-released on process
# exit, so a crashed process never leaves a stuck lock.
LOCK_FILENAME = "state.lock"


def acquire_lock(model_dir, exclusive, timeout=300):
    """Acquire a file lock on model_dir/state.lock.

    shared (exclusive=False): for read-only model loading. Multiple scoring
        processes may hold this concurrently.
    exclusive (exclusive=True): for the state-mutation critical section. Only
        one process may hold it, and not while any shared lock is held.

    Blocks (polling) until acquired or `timeout` seconds elapse. Returns the
    open lock file descriptor; pass it to release_lock().
    """
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, LOCK_FILENAME)
    fd = open(path, "w")
    flag = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    deadline = time.time() + timeout
    while True:
        try:
            fcntl.flock(fd, flag | fcntl.LOCK_NB)
            return fd
        except (BlockingIOError, OSError):
            if time.time() > deadline:
                fd.close()
                kind = "exclusive" if exclusive else "shared"
                raise TimeoutError(
                    f"Timed out waiting for {kind} lock on {path}")
            time.sleep(0.1)


def release_lock(fd):
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        fd.close()


def load_replay_buffer(model_dir):
    path = os.path.join(model_dir, "replay_buffer.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, tuple) and len(data) == 2:
            return data
        return 0, data
    return 0, []


def save_replay_buffer(model_dir, trained_count, buffer):
    # Atomic write: serialize to a temp file then os.replace() into place.
    # Combined with the exclusive lock this guarantees no concurrent reader
    # ever sees a partially-written buffer (and survives a crash mid-write).
    path = os.path.join(model_dir, "replay_buffer.pkl")
    tmp = path + f".tmp.{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump((trained_count, buffer), f)
    os.replace(tmp, path)


def run_optimize(dsn, query, model_dir, optimize_only):
    # Load the model under a SHARED lock so a concurrent training process
    # (which holds the exclusive lock while saving) cannot write model files
    # mid-load. Multiple scoring processes may load in parallel.
    lock_fd = acquire_lock(model_dir, exclusive=False)
    try:
        model = LeroModel(None)
        model.load(model_dir)
    finally:
        release_lock(lock_fd)
    fg = model._feature_generator

    start_time = time.time()

    aliases, edges = extract_join_graph(query)
    table_count = len(aliases)

    if table_count < 2:
        elapsed = time.time() - start_time
        return {
            "optimized_query": query,
            "metadata": {
                "strategy_type": "learning-to-rank",
                "optimization_time": round(elapsed, 6),
                "estimated_impact": 0.0,
                "note": "Single-table query, no optimization needed"
            }
        }

    join_orders = generate_connected_join_orders(aliases, edges)
    candidates = [("", tuple(aliases))] + [
        (f"Leading({' '.join(order)})", order) for order in join_orders
    ]

    # Score all candidates via EXPLAIN (costs only).
    best_hint = ""
    best_score = float("inf")
    default_score = None

    for hint, _ in candidates:
        try:
            plan_json = explain_plan(dsn, query, hint)
            features, _ = fg.transform([plan_json])
            score = float(model.predict(features)[0][0])
            if hint == "":
                default_score = score
            if score < best_score:
                best_score = score
                best_hint = hint
        except Exception:
            continue

    elapsed = time.time() - start_time
    optimized_query = f"/*+ {best_hint} */ {query}" if best_hint else query

    estimated_impact = 0.0
    if default_score is not None and default_score > 0 and best_score < default_score:
        estimated_impact = ((default_score - best_score) / default_score) * 100
    estimated_impact = round(max(0.0, estimated_impact), 2)

    if optimize_only:
        return {
            "optimized_query": optimized_query,
            "metadata": {
                "strategy_type": "learning-to-rank",
                "optimization_time": round(elapsed, 6),
                "estimated_impact": estimated_impact,
                "best_score": round(best_score, 4),
                "num_candidates": len(candidates),
                "mode": "inference-only",
            }
        }

    # Training mode: execute only the best plan via EXPLAIN ANALYZE.
    # This DB execution is read-only w.r.t. local state and may run in
    # parallel with other invocations' scoring/execution.
    try:
        plan_json, best_latency = explain_analyze(dsn, query, best_hint)
    except Exception as e:
        return {
            "optimized_query": query,
            "metadata": {
                "strategy_type": "learning-to-rank",
                "optimization_time": round(time.time() - start_time, 6),
                "estimated_impact": 0.0,
                "error": f"Best plan execution failed: {e}"
            }
        }

    # Critical section: mutate persistent state under an EXCLUSIVE lock so
    # parallel invocations serialize their appends/trains/saves. Reload the
    # replay buffer (and, when training, the model) fresh from disk inside
    # the lock so we build on the latest state rather than the snapshot taken
    # at scoring time — this prevents lost updates when processes overlap.
    lock_fd = acquire_lock(model_dir, exclusive=True)
    try:
        trained_count, replay = load_replay_buffer(model_dir)
        replay.append((best_latency, plan_json))
        if len(replay) > REPLAY_CAP:
            removed = len(replay) - REPLAY_CAP
            replay = replay[-REPLAY_CAP:]
            trained_count = max(0, trained_count - removed)

        untrained = len(replay) - trained_count
        if untrained >= TRAIN_TRIGGER:
            # Reload the model fresh: another process may have fine-tuned and
            # saved since we loaded for scoring.
            train_model = LeroModel(None)
            train_model.load(model_dir)
            train_fg = train_model._feature_generator
            X_train, Y_train = [], []
            for _, plan_json in replay:
                try:
                    features, y_norm = train_fg.transform([plan_json])
                    X_train.append(features[0])
                    Y_train.append(float(y_norm[0]))
                except Exception:
                    continue

            if len(X_train) >= 2:
                try:
                    f = StringIO()
                    with redirect_stdout(f):
                        train_model.fit(X_train, Y_train, pre_training=True)
                    train_model.save(model_dir)
                    trained_count = len(replay)
                except Exception as e:
                    print(f"Warning: model fine-tuning failed: {e}", file=sys.stderr)

        save_replay_buffer(model_dir, trained_count, replay)
    finally:
        release_lock(lock_fd)

    return {
        "optimized_query": optimized_query,
        "metadata": {
            "strategy_type": "learning-to-rank",
            "optimization_time": round(time.time() - start_time, 6),
            "estimated_impact": estimated_impact,
            "best_latency_ms": round(best_latency, 2),
            "num_candidates": len(candidates),
            "mode": "online-training",
            "best_hint": best_hint,
        }
    }


def main():
    args = parse_args()
    model_dir = get_model_dir(args)

    if not os.path.exists(os.path.join(model_dir, "nn_weights")):
        print(json.dumps({
            "error": f"Model not found at {model_dir}. Use --config."
        }))
        sys.exit(1)

    result = run_optimize(args.dsn, args.query, model_dir, args.optimize_only)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
