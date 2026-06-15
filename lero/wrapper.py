#!/usr/bin/env python3
"""Lero Query Optimizer Skill Wrapper.

Generates candidate plans via pg_hint_plan Leading hints with connected join
orders. In training mode, executes plans with EXPLAIN ANALYZE, collects actual
latencies, and fine-tunes the Lero model online. State is persisted across
invocations (model weights, replay buffer).
"""

import argparse
import json
import logging
import os
import pickle
import re
import sys
import time
import warnings
from contextlib import redirect_stdout
from io import StringIO
from itertools import permutations

warnings.filterwarnings("ignore")

import psycopg2

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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "reproduce", "imdb_pw")


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
        plan_obj = result[0]
        return json.dumps(result), plan_obj["Execution Time"]
    finally:
        conn.close()


def extract_join_graph(query):
    q = re.sub(r'--.*?$', '', query, flags=re.MULTILINE)
    q = re.sub(r'/\*.*?\*/', '', q, flags=re.DOTALL)
    q = ' '.join(q.split())

    from_match = re.search(
        r'\bFROM\b\s+(.+?)(?:\bWHERE\b|\bGROUP\b|\bHAVING\b|\bORDER\b|\bLIMIT\b|\bUNION\b|;|$)',
        q, re.IGNORECASE | re.DOTALL
    )
    if not from_match:
        return [], []

    from_clause = from_match.group(1)
    where_clause = ""
    where_match = re.search(
        r'\bWHERE\b\s+(.+?)(?:\bGROUP\b|\bHAVING\b|\bORDER\b|\bLIMIT\b|\bUNION\b|;|$)',
        q, re.IGNORECASE | re.DOTALL
    )
    if where_match:
        where_clause = where_match.group(1)

    parts = []
    depth = 0
    current = ""
    for ch in from_clause:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current.strip())

    aliases = []
    for part in parts:
        if '(' in part:
            m = re.search(r'(?:AS\s+)?(\w+)\s*$', part, re.IGNORECASE)
            if m:
                aliases.append(m.group(1).lower())
            continue
        tokens = part.split()
        if not tokens:
            continue
        if len(tokens) == 1:
            aliases.append(tokens[0].lower())
        elif len(tokens) >= 2:
            if tokens[-2].upper() == 'AS':
                aliases.append(tokens[-1].lower())
            else:
                aliases.append(tokens[-1].lower())

    alias_set = set(aliases)
    edges = set()
    join_pattern = re.findall(
        r'(\w+)\.\w+\s*=\s*(\w+)\.\w+',
        where_clause, re.IGNORECASE
    )
    for a1, a2 in join_pattern:
        a1, a2 = a1.lower(), a2.lower()
        if a1 in alias_set and a2 in alias_set and a1 != a2:
            edge = tuple(sorted([aliases.index(a1), aliases.index(a2)]))
            edges.add(edge)

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


def generate_connected_join_orders(aliases, edges):
    n = len(aliases)
    if n <= 1:
        return [tuple(aliases)]
    if n > 6:
        orders = [tuple(aliases), tuple(reversed(aliases))]
        for start_idx in range(n):
            order = [start_idx]
            remaining = set(range(n)) - {start_idx}
            while remaining:
                found = False
                for r in sorted(remaining):
                    if tuple(sorted([r, order[-1]])) in edges:
                        order.append(r)
                        remaining.remove(r)
                        found = True
                        break
                if not found:
                    r = min(remaining)
                    order.append(r)
                    remaining.remove(r)
            t = tuple(aliases[i] for i in order)
            if t not in orders:
                orders.append(t)
        return orders
    all_orders = []
    for perm in permutations(range(n)):
        if is_connected_join_order(perm, edges):
            all_orders.append(tuple(aliases[i] for i in perm))
    return all_orders


def load_replay_buffer(model_dir):
    path = os.path.join(model_dir, "replay_buffer.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return []


def save_replay_buffer(model_dir, buffer):
    path = os.path.join(model_dir, "replay_buffer.pkl")
    with open(path, "wb") as f:
        pickle.dump(buffer, f)


def run_optimize(dsn, query, model_dir, optimize_only):
    model = LeroModel(None)
    model.load(model_dir)
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

    if optimize_only:
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

        return {
            "optimized_query": optimized_query,
            "metadata": {
                "strategy_type": "learning-to-rank",
                "optimization_time": round(elapsed, 6),
                "estimated_impact": round(max(0.0, estimated_impact), 2),
                "best_score": round(best_score, 4),
                "num_candidates": len(candidates),
                "mode": "inference-only",
            }
        }

    # Training mode: execute plans, collect latencies, train
    plan_data = []

    for hint, _ in candidates:
        try:
            plan_json, latency = explain_analyze(dsn, query, hint)
            plan_data.append((hint, latency, plan_json))
        except Exception as e:
            print(f"Warning: plan execution failed for '{hint or 'default'}': {e}", file=sys.stderr)
            continue

    if not plan_data:
        elapsed = time.time() - start_time
        return {
            "optimized_query": query,
            "metadata": {
                "strategy_type": "learning-to-rank",
                "optimization_time": round(elapsed, 6),
                "estimated_impact": 0.0,
                "error": "No plan could be executed"
            }
        }

    best_entry = min(plan_data, key=lambda x: x[1])
    best_hint = best_entry[0]
    best_latency = best_entry[1]
    default_entry = next((e for e in plan_data if e[0] == ""), plan_data[0])
    default_latency = default_entry[1]

    X_train, Y_train = [], []
    for hint, raw_latency, plan_json in plan_data:
        try:
            features, y_norm = fg.transform([plan_json])
            X_train.append(features[0])
            Y_train.append(float(y_norm[0]))
        except Exception:
            continue

    if len(X_train) >= 2:
        try:
            f = StringIO()
            with redirect_stdout(f):
                model.fit(X_train, Y_train, pre_training=True)
            model.save(model_dir)
        except Exception as e:
            print(f"Warning: model fine-tuning failed: {e}", file=sys.stderr)

    replay = load_replay_buffer(model_dir)
    for hint, latency, plan_json in plan_data:
        replay.append((latency, plan_json))
    if len(replay) > 1000:
        replay = replay[-1000:]
    save_replay_buffer(model_dir, replay)

    elapsed = time.time() - start_time

    optimized_query = f"/*+ {best_hint} */ {query}" if best_hint else query
    estimated_impact = 0.0
    if default_latency > 0 and best_latency < default_latency:
        estimated_impact = ((default_latency - best_latency) / default_latency) * 100

    metadata = {
        "strategy_type": "learning-to-rank",
        "optimization_time": round(elapsed, 6),
        "estimated_impact": round(max(0.0, estimated_impact), 2),
        "best_latency_ms": round(best_latency, 2),
        "default_latency_ms": round(default_latency, 2),
        "num_candidates": len(plan_data),
        "mode": "online-training",
    }
    if best_hint:
        metadata["best_hint"] = best_hint

    return {
        "optimized_query": optimized_query,
        "metadata": metadata
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
