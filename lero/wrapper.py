#!/usr/bin/env python3
"""Lero Query Optimizer Skill — single-file long-lived HTTP server wrapper.

Generates candidate plans via pg_hint_plan Leading hints with connected join
orders, scores them with the Lero model, and emits the best as a hinted query.

Design:
  * The optimize critical path is lock-free: scoring reads an atomic
    (model, fg, mtime) snapshot. State (model/replay/counters) lives in memory
    and is persisted for restart recovery.
  * In online-training mode, EXPLAIN ANALYZE (latency collection) runs in a
    background thread, so the response returns immediately.
  * When enough new samples accumulate, a separate *process* is spawned
    (`wrapper.py --train --model-dir ...`) to fit the model and save it
    atomically; the server picks up the new weights via an mtime check on the
    next request (lock-free snapshot swap).

Endpoints: GET /health, POST /optimize, GET /state, POST /shutdown.
"""

import argparse
import json
import os
import pickle
import queue as queue_mod
import random
import signal
import subprocess
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from itertools import permutations

warnings.filterwarnings("ignore")

import psycopg2
import sqlglot
from sqlglot import exp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import LeroModel

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reproduce", "imdb_pw")
REPLAY_CAP_DEFAULT = 10000
TRAIN_TRIGGER_DEFAULT = 100
CANDIDATE_LIMIT_DEFAULT = 100
POOL_SIZE_DEFAULT = 4
# Safe-gate: candidates whose cost is more than this many times the default
# plan's cost are dropped before scoring (don't consider a plan that would
# take minutes to execute), and background EXPLAIN ANALYZE / training is
# skipped when the chosen plan is catastrophically costlier than the
# default. The cold-start model occasionally selects such plans.
_MAX_COST_RATIO = 50.0


def _plan_total_cost(plan_json_str):
    """Total Cost from an EXPLAIN (COSTS TRUE, FORMAT JSON) plan (a JSON
    string as produced by _explain_plan), or None."""
    try:
        node = json.loads(plan_json_str)
        return float(node[0]["Plan"]["Total Cost"])
    except Exception:
        return None


def _cost_ratio_too_high(chosen_cost, default_cost, max_ratio=_MAX_COST_RATIO):
    """True if the chosen plan's cost is catastrophically higher than the
    default's. Unknown / non-positive costs → False (let it through)."""
    if not chosen_cost or not default_cost:
        return False
    try:
        d = float(default_cost)
        if d <= 0:
            return False
        return float(chosen_cost) / d > max_ratio
    except (TypeError, ValueError, ZeroDivisionError):
        return False


# --------------------------------------------------------------------------- #
# DB connection pool: reuses psycopg2 connections across EXPLAIN calls.
# --------------------------------------------------------------------------- #
class ConnectionPool:
    def __init__(self, size=POOL_SIZE_DEFAULT):
        self.size = size
        self._pools = {}
        self._created = {}
        self._lock = threading.Lock()

    def _queue(self, dsn):
        with self._lock:
            q = self._pools.get(dsn)
            if q is None:
                q = queue_mod.Queue(maxsize=self.size)
                self._pools[dsn] = q
                self._created[dsn] = 0
            return q

    def get(self, dsn):
        q = self._queue(dsn)
        try:
            return q.get_nowait()
        except queue_mod.Empty:
            pass
        with self._lock:
            if self._created[dsn] < self.size:
                self._created[dsn] += 1
                conn = psycopg2.connect(dsn)
                conn.autocommit = True
                conn.set_client_encoding("UTF8")
                return conn
        return q.get(timeout=60)

    def put(self, dsn, conn):
        self._queue(dsn).put(conn)

    def discard(self, dsn, conn):
        """Close a broken connection and free its slot so the pool can grow."""
        self._queue(dsn)
        with self._lock:
            try:
                conn.close()
            except Exception:
                pass
            if self._created.get(dsn, 0) > 0:
                self._created[dsn] -= 1

    def close_all(self):
        with self._lock:
            for q in self._pools.values():
                while True:
                    try:
                        conn = q.get_nowait()
                    except queue_mod.Empty:
                        break
                    try:
                        conn.close()
                    except Exception:
                        pass
            self._pools.clear()
            self._created.clear()


# --------------------------------------------------------------------------- #
# Join-graph extraction and candidate generation (schema-agnostic).
# --------------------------------------------------------------------------- #
def extract_join_graph(query):
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


# --------------------------------------------------------------------------- #
# Training subprocess entry point: load model + replay from disk, fit, save
# atomically. Runs in its own process so it never blocks the server's threads
# (separate GIL). The server detects the new weights via an mtime check.
# --------------------------------------------------------------------------- #
def run_training(model_dir):
    nn_path = os.path.join(model_dir, "nn_weights")
    if not os.path.exists(nn_path):
        print(f"model not found at {model_dir}", file=sys.stderr)
        return 1
    replay_path = os.path.join(model_dir, "replay_buffer.pkl")
    if not os.path.exists(replay_path):
        return 0

    model = LeroModel(None)
    model.load(model_dir)
    fg = model._feature_generator

    with open(replay_path, "rb") as f:
        data = pickle.load(f)
    replay = data[1] if isinstance(data, tuple) and len(data) == 2 else data
    if len(replay) < 2:
        return 0

    X, Y = [], []
    for _, pj in replay:
        try:
            features, y_norm = fg.transform([pj])
            X.append(features[0])
            Y.append(float(y_norm[0]))
        except Exception:
            continue
    if len(X) < 2:
        return 0

    # Save to a temp dir, then os.replace each file into model_dir so readers
    # never observe a partial write.
    tmp_dir = f"{model_dir}.tmp.{os.getpid()}"
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        with redirect_stdout(StringIO()):
            model.fit(X, Y, pre_training=True)
        model.save(tmp_dir)
        for fn in os.listdir(tmp_dir):
            os.replace(os.path.join(tmp_dir, fn), os.path.join(model_dir, fn))
    except Exception as e:
        print(f"training fit failed: {e}", file=sys.stderr)
        return 1
    finally:
        try:
            for fn in os.listdir(tmp_dir):
                p = os.path.join(tmp_dir, fn)
                if os.path.exists(p):
                    os.remove(p)
            os.rmdir(tmp_dir)
        except OSError:
            pass
    return 0


# --------------------------------------------------------------------------- #
# Skill: holds all in-memory state; one instance per server process.
# --------------------------------------------------------------------------- #
class LeroSkill:
    def __init__(self):
        self.pool = ConnectionPool(POOL_SIZE_DEFAULT)
        # Separate, smaller pool for background training-data collection so
        # long-running EXPLAIN ANALYZE never starves foreground scoring.
        self._bg_pool = ConnectionPool(max(1, POOL_SIZE_DEFAULT // 2))
        self._snapshot = (None, None, 0)  # (model, fg, nn_weights mtime)
        self.model_dir = None
        self.replay = []
        self.trained_count = 0
        self.replay_cap = REPLAY_CAP_DEFAULT
        self.train_trigger = TRAIN_TRIGGER_DEFAULT
        self.candidate_limit = CANDIDATE_LIMIT_DEFAULT
        # _state_lock guards replay-buffer append/persist + the training-spawn
        # flag. It is never held on the optimize critical path.
        self._state_lock = threading.Lock()
        self._training_in_progress = False
        self._collecting_in_progress = False  # True while the bg thread is mid-EXPLAIN-ANALYZE collection
        self._train_proc = None  # Popen of the in-flight training subprocess
        self._shutting_down = False  # set by persist(); /optimize then 503s
        self._persist_lock = threading.Lock()  # single-flight: concurrent /shutdown runs persist once
        self._bg_queue = queue_mod.Queue()
        self._bg_thread = threading.Thread(target=self._bg_loop, daemon=True)
        self._bg_thread.start()

    def _apply_config(self, config):
        self.replay_cap = int(config.get("replay_cap", REPLAY_CAP_DEFAULT))
        self.train_trigger = int(config.get("train_trigger", TRAIN_TRIGGER_DEFAULT))
        self.candidate_limit = int(config.get("candidate_limit", CANDIDATE_LIMIT_DEFAULT))
        pool_size = int(config.get("connection_pool_size", POOL_SIZE_DEFAULT))
        if self.pool.size != pool_size:
            self.pool.close_all()
            self.pool = ConnectionPool(pool_size)

    def ensure_loaded(self, config):
        model_dir = config.get("model_dir") or DEFAULT_MODEL_DIR
        self._apply_config(config)
        if self._snapshot[0] is None or model_dir != self.model_dir:
            self._load_initial(model_dir)

    def _load_initial(self, model_dir):
        nn_path = os.path.join(model_dir, "nn_weights")
        if not os.path.exists(nn_path):
            raise FileNotFoundError(f"Model not found at {model_dir}. Set --config model_dir.")
        # Load model + replay into fresh objects, then publish atomically.
        model = LeroModel(None)
        model.load(model_dir)
        fg = model._feature_generator
        mtime = os.path.getmtime(nn_path)
        replay, trained_count = self._load_replay(model_dir)
        self.model_dir = model_dir
        self.replay = replay
        self.trained_count = trained_count
        self._snapshot = (model, fg, mtime)

    def _maybe_reload_model(self):
        """Lock-free: if nn_weights mtime changed, load a fresh model and swap
        the snapshot. A scoring call that already captured the old snapshot is
        unaffected (it holds its own ref)."""
        if self.model_dir is None:
            return
        try:
            mtime = os.path.getmtime(os.path.join(self.model_dir, "nn_weights"))
        except OSError:
            return
        if mtime == self._snapshot[2]:
            return
        try:
            model = LeroModel(None)
            model.load(self.model_dir)
            self._snapshot = (model, model._feature_generator, mtime)
        except Exception as e:
            print(f"model reload failed: {e}", file=sys.stderr)

    def _load_replay(self, model_dir):
        path = os.path.join(model_dir, "replay_buffer.pkl")
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, tuple) and len(data) == 2:
                    return list(data[1]), data[0]
                return list(data), 0
            except Exception:
                pass
        return [], 0

    def _save_replay(self):
        path = os.path.join(self.model_dir, "replay_buffer.pkl")
        tmp = path + f".tmp.{os.getpid()}"
        with open(tmp, "wb") as f:
            pickle.dump((self.trained_count, self.replay), f)
        os.replace(tmp, path)

    # -- DB helpers (pooled) --
    def _explain_plan(self, dsn, query, hint_str=""):
        conn = self.pool.get(dsn)
        ok = False
        try:
            cur = conn.cursor()
            sql = query.strip().rstrip(';').strip()
            if hint_str:
                sql = f"/*+ {hint_str} */ EXPLAIN (COSTS TRUE, FORMAT JSON) {sql}"
            else:
                sql = f"EXPLAIN (COSTS TRUE, FORMAT JSON) {sql}"
            cur.execute(sql)
            result = cur.fetchone()[0]
            cur.close()
            if isinstance(result, list) and len(result) == 2:
                result = [result[1]]
            ok = True
            return json.dumps(result)
        finally:
            if ok:
                try:
                    conn.rollback()
                except Exception:
                    pass
                self.pool.put(dsn, conn)
            else:
                self.pool.discard(dsn, conn)

    def _explain_analyze(self, dsn, query, hint_str="", pool=None):
        pool = pool or self.pool
        conn = pool.get(dsn)
        ok = False
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
            cur.close()
            if isinstance(result, list) and len(result) == 2:
                result = [result[1]]
            ok = True
            return json.dumps(result), result[0]["Execution Time"]
        finally:
            if ok:
                try:
                    conn.rollback()
                except Exception:
                    pass
                pool.put(dsn, conn)
            else:
                pool.discard(dsn, conn)

    # -- Critical path (lock-free) --
    def optimize(self, dsn, query, optimize_only):
        self._maybe_reload_model()
        start_time = time.time()

        model, fg, _ = self._snapshot
        if model is None:
            return {
                "optimized_query": query,
                "metadata": {
                    "strategy_type": "learning-to-rank",
                    "optimization_time": round(time.time() - start_time, 6),
                    "estimated_impact": 0.0,
                    "error": "model not loaded",
                },
            }

        aliases, edges = extract_join_graph(query)
        if len(aliases) < 2:
            return {
                "optimized_query": query,
                "metadata": {
                    "strategy_type": "learning-to-rank",
                    "optimization_time": round(time.time() - start_time, 6),
                    "estimated_impact": 0.0,
                    "note": "Single-table query, no optimization needed",
                    "mode": "inference-only" if optimize_only else "online-training",
                },
            }

        join_orders = generate_connected_join_orders(aliases, edges, self.candidate_limit)
        candidates = [("", tuple(aliases))] + [
            (f"Leading({' '.join(order)})", order) for order in join_orders
        ]

        # Collect all candidate plans in parallel (cost-only EXPLAIN).
        hints = [c[0] for c in candidates]
        workers = max(1, self.pool.size)
        plan_jsons = [None] * len(hints)

        def _one(i):
            try:
                plan_jsons[i] = self._explain_plan(dsn, query, hints[i])
            except Exception:
                plan_jsons[i] = None

        if len(hints) <= 1:
            for i in range(len(hints)):
                _one(i)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                list(ex.map(_one, range(len(hints))))

        # Batched featurize + predict.
        best_hint = ""
        best_idx = -1
        best_score = float("inf")
        default_score = None
        valid_idx = [i for i, pj in enumerate(plan_jsons) if pj is not None]
        # Safe-gate: the default plan is candidates[0] (no hint). Drop
        # candidates whose cost is catastrophically higher than the default's
        # so the model can't pick a plan that would take minutes to execute.
        default_cost = _plan_total_cost(plan_jsons[0]) if plan_jsons[0] is not None else None
        if default_cost and default_cost > 0:
            valid_idx = [i for i in valid_idx
                         if not _cost_ratio_too_high(_plan_total_cost(plan_jsons[i]), default_cost)]
        if valid_idx:
            try:
                features, _ = fg.transform([plan_jsons[i] for i in valid_idx])
                scores = model.predict(features)
                for j, i in enumerate(valid_idx):
                    try:
                        score = float(scores[j][0])
                    except Exception:
                        continue
                    if hints[i] == "":
                        default_score = score
                    if score < best_score:
                        best_score = score
                        best_hint = hints[i]
                        best_idx = i
            except Exception as e:
                print(f"batched scoring failed: {e}", file=sys.stderr)

        optimized_query = f"/*+ {best_hint} */ {query}" if best_hint else query
        estimated_impact = 0.0
        if default_score is not None and default_score > 0 and best_score < default_score:
            estimated_impact = ((default_score - best_score) / default_score) * 100
        estimated_impact = round(max(0.0, estimated_impact), 2)

        mode = "inference-only" if optimize_only else "online-training"
        result = {
            "optimized_query": optimized_query,
            "metadata": {
                "strategy_type": "learning-to-rank",
                "optimization_time": round(time.time() - start_time, 6),
                "estimated_impact": estimated_impact,
                "best_score": round(best_score, 4) if best_score != float("inf") else None,
                "num_candidates": len(candidates),
                "mode": mode,
                "best_hint": best_hint,
            },
        }

        if not optimize_only and best_hint is not None:
            # Training-data collection (EXPLAIN ANALYZE) runs in the background;
            # the response returns immediately. Safe-gate: skip if the chosen
            # plan is catastrophically costlier than the default (would hang
            # the background thread on a doomed execution).
            chosen_cost = (_plan_total_cost(plan_jsons[best_idx])
                           if best_idx >= 0 and best_idx < len(plan_jsons)
                           and plan_jsons[best_idx] is not None else None)
            if not _cost_ratio_too_high(chosen_cost, default_cost):
                self._bg_queue.put(("collect", dsn, query, best_hint))

        return result

    # -- Background training-data collection + training spawn --
    def _bg_loop(self):
        while True:
            task = self._bg_queue.get()
            with self._state_lock:
                self._collecting_in_progress = True
            try:
                self._collect_training_data(task)
            except Exception as e:
                print(f"bg training-data collection failed: {e}", file=sys.stderr)
            finally:
                with self._state_lock:
                    self._collecting_in_progress = False

    def _collect_training_data(self, task):
        _, dsn, query, best_hint = task
        try:
            plan_json, best_latency = self._explain_analyze(dsn, query, best_hint, pool=self._bg_pool)
        except Exception as e:
            print(f"EXPLAIN ANALYZE failed: {e}", file=sys.stderr)
            return

        spawn = False
        with self._state_lock:
            self.replay.append((best_latency, plan_json))
            if len(self.replay) > self.replay_cap:
                removed = len(self.replay) - self.replay_cap
                self.replay = self.replay[-self.replay_cap:]
                self.trained_count = max(0, self.trained_count - removed)
            untrained = len(self.replay) - self.trained_count
            if untrained >= self.train_trigger and not self._training_in_progress:
                self._training_in_progress = True
                self.trained_count = len(self.replay)
                spawn = True
            self._save_replay()
        if spawn:
            self._spawn_training_worker()

    def _spawn_training_worker(self):
        model_dir = self.model_dir

        def _run():
            try:
                p = subprocess.Popen(
                    [sys.executable, os.path.abspath(__file__), "--train", "--model-dir", model_dir]
                )
                with self._state_lock:
                    self._train_proc = p
                p.wait()
            except Exception as e:
                print(f"training worker failed: {e}", file=sys.stderr)
            finally:
                with self._state_lock:
                    self._training_in_progress = False
                    self._train_proc = None

        threading.Thread(target=_run, daemon=True).start()

    def _wait_for_training(self, timeout=300):
        """Block until the in-flight training subprocess finishes (its model
        save completes) or the timeout expires."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._state_lock:
                in_progress = self._training_in_progress
            if not in_progress:
                return
            time.sleep(0.5)

    def persist(self):
        if not self._persist_lock.acquire(blocking=False):
            # Another persist() is in flight (concurrent /shutdown or SIGTERM
            # racing a skill_runner stop_wrapper); it will complete the save.
            # Returning here avoids two in-process run_training calls
            # clobbering the same pid-based temp dir.
            return
        try:
                # Don't bypass the model save on shutdown. The replay buffer holds
                # untrained samples in memory; save it first so the training worker /
                # restart can rehydrate. Then wait for any in-flight training
                # subprocess so its model save isn't abandoned, and if samples remain
                # untrained, run a final in-process training so the persisted model
                # reflects all collected data.
                self._shutting_down = True
                with self._state_lock:
                    untrained = len(self.replay) - self.trained_count
                    if self.model_dir is not None:
                        self._save_replay()
                self._wait_for_training(timeout=300)
                if self.model_dir is not None and untrained > 0:
                    try:
                        run_training(self.model_dir)
                    except Exception as e:
                        print(f"final training on shutdown failed: {e}", file=sys.stderr)
        finally:
            self._persist_lock.release()
            # Only the persist that acquired the lock shuts down the
            # server — a guarded-out persist (concurrent /shutdown)
            # must NOT start SERVER.shutdown, or daemon_threads=True
            # would let main() exit and kill this in-flight save.
            if SERVER is not None:
                threading.Thread(target=SERVER.shutdown, daemon=True).start()

    def state_summary(self):
        model = self._snapshot[0]
        with self._state_lock:
            # `idle`: True only when, with no new prompts arriving, this skill
            # does no background work — no training subprocess, no in-flight bg
            # EXPLAIN ANALYZE, nothing queued. Below-trigger untrained samples
            # don't count (no activity until the next foreground call).
            idle = (not self._training_in_progress
                    and not self._collecting_in_progress
                    and self._bg_queue.qsize() == 0)
            return {
                "replay_buffer_len": len(self.replay),
                "trained_count": self.trained_count,
                "model_dir": self.model_dir,
                "pool_size": self.pool.size,
                "loaded": model is not None,
                "idle": idle,
            }


# --------------------------------------------------------------------------- #
# HTTP server.
# --------------------------------------------------------------------------- #
SERVER = None
SKILL = LeroSkill()


def _drain_and_stop():
    try:
        SKILL.persist()  # persist() starts SERVER.shutdown in its finally (only if it owns the lock)
    except Exception as e:
        print(f"persist failed: {e}", file=sys.stderr)
        if SERVER is not None:
            threading.Thread(target=SERVER.shutdown, daemon=True).start()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send_json(200, {"status": "ok"})
        elif self.path.startswith("/state"):
            self._send_json(200, SKILL.state_summary())
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/optimize":
            if SKILL._shutting_down:
                self._send_json(503, {"error": "server is shutting down"})
                return
            try:
                n = int(self.headers.get("Content-Length", "0") or "0")
                raw = self.rfile.read(n).decode() if n else "{}"
                req = json.loads(raw) if raw else {}
            except Exception as e:
                self._send_json(400, {"error": f"bad request body: {e}"})
                return
            dsn = req.get("dsn")
            query = req.get("query")
            optimize_only = bool(req.get("optimize_only"))
            cfg = req.get("config")
            if isinstance(cfg, str) and cfg:
                try:
                    cfg = json.loads(cfg)
                except Exception:
                    cfg = {"model_dir": cfg}
            cfg = cfg or {}
            if not isinstance(cfg, dict):
                self._send_json(400, {"error": "config must be a JSON object"})
                return
            if not dsn or not query:
                self._send_json(400, {"error": "dsn and query are required"})
                return
            try:
                SKILL.ensure_loaded(cfg)
                result = SKILL.optimize(dsn, query, optimize_only)
            except FileNotFoundError as e:
                self._send_json(400, {"error": str(e)})
            except Exception as e:
                self._send_json(500, {"error": str(e)})
            else:
                self._send_json(200, result)
        elif self.path == "/shutdown":
            self._send_json(200, {"status": "shutting down"})
            _drain_and_stop()
        else:
            self._send_json(404, {"error": "not found"})


def _signal_handler(signum, frame):
    _drain_and_stop()


def main():
    parser = argparse.ArgumentParser(description="Lero Query Optimizer server")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "0")))
    parser.add_argument("--train", action="store_true", help="Run a single training cycle and exit (subprocess mode).")
    parser.add_argument("--model-dir", default=None, help="Model directory (for --train mode).")
    args = parser.parse_args()

    if args.train:
        sys.exit(run_training(args.model_dir or DEFAULT_MODEL_DIR))

    if not args.port:
        print("error: --port or PORT env required", file=sys.stderr)
        sys.exit(1)

    global SERVER
    SERVER = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    print(f"Lero skill server listening on 127.0.0.1:{args.port}", flush=True)
    try:
        SERVER.serve_forever()
    finally:
        SKILL.pool.close_all()
        SKILL._bg_pool.close_all()


if __name__ == "__main__":
    main()
