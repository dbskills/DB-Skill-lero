#!/usr/bin/env python3
"""
Lero Query Optimizer Skill Wrapper
Uses the Lero model to score candidate plans generated via pg_hint_plan
Leading hints with connected join orders. Uses PostgreSQL cost estimates
as primary filter and Lero model score as tiebreaker.
"""

import argparse
import json
import os
import re
import sys
import time
from itertools import permutations

import psycopg2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import LeroModel


def parse_args():
    parser = argparse.ArgumentParser(description="Lero Query Optimizer")
    parser.add_argument("--dsn", required=True, help="Database connection string")
    parser.add_argument("--query", required=True, help="SQL query to optimize")
    parser.add_argument("--config", required=False, help="Model configuration path")
    parser.add_argument("--optimize-only", action="store_true",
                        help="Optimize-only mode: bypass actual query execution")
    return parser.parse_args()


def get_explain_plan(dsn, query, hint_str=""):
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
        can_join = False
        for j in joined:
            if tuple(sorted([t, j])) in edges:
                can_join = True
                break
        if not can_join:
            return False
        joined.add(t)
    return True


def generate_connected_join_orders(aliases, edges):
    n = len(aliases)
    if n <= 1:
        return [tuple(aliases)]

    if n > 6:
        orders = [tuple(aliases)]
        orders.append(tuple(reversed(aliases)))
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


def run_optimize(dsn, query, model_path, optimize_only):
    model = LeroModel(None)
    model.load(model_path)
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

    # Evaluate all candidates
    candidates = []

    # Default plan
    try:
        plan_json = get_explain_plan(dsn, query)
        plan_obj = json.loads(plan_json)
        default_cost = plan_obj[0]['Plan']['Total Cost']
        features, _ = fg.transform([plan_json])
        pred = model.predict(features)
        default_score = float(pred[0][0])
        candidates.append({
            'hint': '',
            'cost': default_cost,
            'score': default_score,
            'plan_json': plan_json
        })
    except Exception as e:
        print(f"Warning: default plan evaluation failed: {e}", file=sys.stderr)
        default_cost = float('inf')
        default_score = float('inf')

    # Evaluate Leading hints
    for order in join_orders:
        hint = f"Leading({' '.join(order)})"
        try:
            plan_json = get_explain_plan(dsn, query, hint)
            plan_obj = json.loads(plan_json)
            features, _ = fg.transform([plan_json])
            pred = model.predict(features)
            candidates.append({
                'hint': hint,
                'cost': plan_obj[0]['Plan']['Total Cost'],
                'score': float(pred[0][0]),
                'plan_json': plan_json
            })
        except Exception:
            continue

    if not candidates:
        elapsed = time.time() - start_time
        return {
            "optimized_query": query,
            "metadata": {
                "strategy_type": "learning-to-rank",
                "optimization_time": round(elapsed, 6),
                "estimated_impact": 0.0,
                "error": "No candidate plans could be evaluated"
            }
        }

    # Filter candidates: only keep those with cost <= default_cost * 1.1
    if default_cost != float('inf'):
        cost_threshold = default_cost * 1.1
        viable = [c for c in candidates if c['cost'] <= cost_threshold]
    else:
        viable = candidates

    if not viable:
        viable = candidates  # fallback: use all if none are viable

    # Among viable candidates, select the one with lowest model score
    best = min(viable, key=lambda c: c['score'])

    elapsed = time.time() - start_time

    if best['hint']:
        optimized_query = f"/*+ {best['hint']} */ {query}"
    else:
        optimized_query = query

    estimated_impact = 0.0
    if default_score and default_score > 0 and best['score'] < default_score:
        estimated_impact = ((default_score - best['score']) / default_score) * 100

    metadata = {
        "strategy_type": "learning-to-rank",
        "optimization_time": round(elapsed, 6),
        "estimated_impact": round(max(0.0, estimated_impact), 2),
        "best_score": round(best['score'], 4),
        "num_candidates": len(candidates),
        "num_viable": len(viable),
    }
    if default_score != float('inf'):
        metadata["default_score"] = round(default_score, 4)
    if best['hint']:
        metadata["best_hint"] = best['hint']

    return {
        "optimized_query": optimized_query,
        "metadata": metadata
    }


def main():
    args = parse_args()

    model_path = args.config
    if not model_path:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "reproduce", "imdb_pw")

    if not os.path.exists(os.path.join(model_path, "nn_weights")):
        print(json.dumps({
            "error": f"Model not found at {model_path}. Use --config."
        }))
        sys.exit(1)

    result = run_optimize(args.dsn, args.query, model_path, args.optimize_only)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
