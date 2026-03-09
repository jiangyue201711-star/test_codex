"""PRD-driven synthetic data pipeline (stdlib-only)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass
class DomainSpec:
    name: str
    ratio: float
    task_types: List[str]
    params: Dict[str, Any]


def choose_domain(domains: List[DomainSpec], rng: random.Random) -> DomainSpec:
    total = sum(d.ratio for d in domains)
    x = rng.random() * total
    acc = 0.0
    for d in domains:
        acc += d.ratio
        if x <= acc:
            return d
    return domains[-1]


def mat_vec(A: List[List[float]], v: List[float]) -> List[float]:
    return [sum(aij * vj for aij, vj in zip(row, v)) for row in A]


def vec_norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def estimate_dominant_eigenvalue(A: List[List[float]], iters: int = 20) -> Tuple[float, float]:
    n = len(A)
    v = [1.0] * n
    lam = 0.0
    for _ in range(iters):
        Av = mat_vec(A, v)
        denom = sum(x * x for x in v) or 1.0
        lam = sum(a * b for a, b in zip(Av, v)) / denom
        nv = vec_norm(Av) or 1.0
        v = [x / nv for x in Av]
    Av = mat_vec(A, v)
    residual = vec_norm([a - lam * b for a, b in zip(Av, v)])
    return lam, residual


def matrix_sample(spec: DomainSpec, rng: random.Random) -> Dict[str, Any]:
    n = rng.randint(spec.params.get("n_min", 2), spec.params.get("n_max", 10))
    family = rng.choice(spec.params.get("families", ["diagonal_dominant"]))
    A = [[rng.uniform(-1.0, 1.0) for _ in range(n)] for _ in range(n)]
    if family == "diagonal_dominant":
        for i in range(n):
            A[i][i] += n
    else:  # nonsym_scaled
        scales = [rng.uniform(0.3, 2.0) for _ in range(n)]
        for i in range(n):
            for j in range(n):
                A[i][j] *= scales[i]

    lam, residual = estimate_dominant_eigenvalue(A)
    return {
        "instruction": "实现并优化 find_dominant_eigenvalue_and_eigenvector(A)",
        "inputs": {"A": A},
        "reference_output": {"dominant_eigenvalue_estimate": lam, "residual_estimate": residual},
        "constraints": {"dominant_by_magnitude": True},
        "metrics": {"correctness": "allclose_like", "performance": "median_latency"},
        "metadata": {"n": n, "family": family}
    }


def string_sample(spec: DomainSpec, rng: random.Random) -> Dict[str, Any]:
    lines = rng.randint(20, spec.params.get("max_lines", 200))
    dirty_rate = spec.params.get("dirty_rate", 0.1)
    data, good = [], 0
    for i in range(lines):
        if rng.random() < dirty_rate:
            data.append(f"BAD::{i}")
        else:
            data.append(f"ts=2026-01-01 level=INFO id={1000+i}")
            good += 1
    return {
        "instruction": "解析日志并统计有效INFO行",
        "inputs": {"lines": data},
        "reference_output": {"valid_info_count": good},
        "constraints": {"robust_to_dirty": True},
        "metrics": {"correctness": "exact_match", "performance": "records_per_second"}
    }


def graph_sample(spec: DomainSpec, rng: random.Random) -> Dict[str, Any]:
    n = rng.randint(spec.params.get("n_min", 20), spec.params.get("n_max", 80))
    d0, d1 = spec.params.get("density", [0.05, 0.2])
    density = rng.uniform(d0, d1)
    INF = 10**12
    dist = [[INF] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < density:
                w = rng.randint(1, 20)
                dist[i][j] = dist[j][i] = w
                edges.append((i, j, w))
    for k in range(n):
        dk = dist[k]
        for i in range(n):
            dik = dist[i][k]
            row = dist[i]
            for j in range(n):
                nd = dik + dk[j]
                if nd < row[j]:
                    row[j] = nd
    best = dist[0][n - 1]
    if best >= INF:
        best = None
    return {
        "instruction": "计算从0到n-1的最短路径代价",
        "inputs": {"n": n, "edges": edges, "src": 0, "dst": n - 1},
        "reference_output": {"shortest_cost": best},
        "constraints": {"non_negative_weight": True},
        "metrics": {"correctness": "exact_or_none", "performance": "median_latency"}
    }


def db_sample(spec: DomainSpec, rng: random.Random) -> Dict[str, Any]:
    lo, hi = spec.params.get("rows", [1000, 5000])
    rows = rng.randint(lo, hi)
    null_rate = spec.params.get("null_rate", 0.05)
    table, total = [], 0.0
    for _ in range(rows):
        if rng.random() < null_rate:
            amt = None
        else:
            amt = round(rng.uniform(1, 500), 2)
            total += amt
        table.append({"amount": amt})
    return {
        "instruction": "过滤非空amount并求和",
        "inputs": {"table": table},
        "reference_output": {"sum_amount": round(total, 2)},
        "constraints": {"precision": 2},
        "metrics": {"correctness": "exact_match", "performance": "rows_per_second"}
    }


def system_sample(spec: DomainSpec, rng: random.Random) -> Dict[str, Any]:
    lo, hi = spec.params.get("jobs", [50, 300])
    jobs = rng.randint(lo, hi)
    fail_rate = spec.params.get("fail_rate", 0.03)
    durations = [rng.randint(1, 30) for _ in range(jobs)]
    failed = sum(1 for _ in range(jobs) if rng.random() < fail_rate)
    return {
        "instruction": "统计批任务总时长与失败数量",
        "inputs": {"durations": durations, "fail_rate": fail_rate},
        "reference_output": {"total_time": sum(durations), "failed_jobs": failed},
        "constraints": {"no_drop": True},
        "metrics": {"correctness": "exact_match", "reliability": "recovery_rate"}
    }


def api_code_sample(spec: DomainSpec, rng: random.Random) -> Dict[str, Any]:
    t0, t1 = spec.params.get("timeout_ms", [100, 1200])
    timeout = rng.randint(t0, t1)
    lats = [rng.randint(50, 300) for _ in range(rng.randint(3, 8))]
    return {
        "instruction": "在超时约束下给出可行API调度计划",
        "inputs": {"timeout_ms": timeout, "latencies_ms": lats},
        "reference_output": {"sum_latency_ms": sum(lats), "is_feasible": sum(lats) <= timeout},
        "constraints": {"respect_timeout": True},
        "metrics": {"correctness": "feasible_plan", "cost": "api_calls"}
    }


def build_sample(domain: DomainSpec, sid: int, rng: random.Random) -> Dict[str, Any]:
    fn = {
        "matrix": matrix_sample,
        "string_parsing": string_sample,
        "graph": graph_sample,
        "db_retrieval": db_sample,
        "system_concurrency": system_sample,
        "api_code": api_code_sample,
    }[domain.name]
    s = fn(domain, rng)
    s["task_id"] = f"{domain.name}-{sid:06d}"
    s["domain"] = domain.name
    s["difficulty"] = rng.choice(["easy", "medium", "hard"])
    return s


def split_data(items: List[Dict[str, Any]], split: Dict[str, float]) -> Tuple[List[Any], List[Any], List[Any]]:
    n = len(items)
    n_train = int(n * split["train"])
    n_val = int(n * split["val"])
    return items[:n_train], items[n_train : n_train + n_val], items[n_train + n_val :]


def dump_jsonl(path: Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    seed = int(cfg["generation"]["random_seed"])
    total = int(cfg["generation"]["total_samples"])
    rng = random.Random(seed)

    domains = [DomainSpec(**d) for d in cfg["domains"]]
    items = []
    for i in range(total):
        d = choose_domain(domains, rng)
        items.append(build_sample(d, i, rng))
    rng.shuffle(items)

    train, val, test = split_data(items, cfg["split"])
    root = Path(cfg["output"]["root"])
    dump_jsonl(root / "train.jsonl", train)
    dump_jsonl(root / "val.jsonl", val)
    dump_jsonl(root / "test.jsonl", test)

    manifest = {
        "version": cfg["version"],
        "project": cfg["project"],
        "total_samples": total,
        "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "seed": seed,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    stats: Dict[str, int] = {}
    for x in items:
        stats[x["domain"]] = stats.get(x["domain"], 0) + 1
    with (root / "domain_stats.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["domain", "count"])
        for k in sorted(stats):
            w.writerow([k, stats[k]])

    print(f"generated {total} samples -> {root}")


if __name__ == "__main__":
    main()
