"""PRD-driven synthetic data pipeline (stdlib-only).

Focus: generate higher-fidelity dominant-eigenpair tasks aligned to:
- /app/eigen.py entrypoint: find_dominant_eigenvalue_and_eigenvector
- /app/eval.py median-latency evaluation with allclose correctness
"""

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
    if total <= 0:
        raise ValueError("sum of domain ratios must be > 0")
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


def estimate_dominant_eigenvalue(A: List[List[float]], iters: int = 30) -> Tuple[float, float]:
    """Power-iteration estimate (real-only proxy label; good enough for synthetic task metadata)."""
    n = len(A)
    v = [1.0 / math.sqrt(n)] * n
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


def _build_matrix(n: int, family: str, rng: random.Random) -> List[List[float]]:
    A = [[rng.uniform(-1.0, 1.0) for _ in range(n)] for _ in range(n)]

    if family == "diagonal_dominant":
        for i in range(n):
            A[i][i] += n + rng.uniform(0.0, 1.0)
        return A

    if family == "nonsym_scaled":
        scales = [rng.uniform(0.2, 2.5) for _ in range(n)]
        for i in range(n):
            for j in range(n):
                A[i][j] *= scales[i]
        return A

    if family == "near_defective":
        # Jordan-like upper band + tiny perturbation
        A = [[0.0 for _ in range(n)] for _ in range(n)]
        base = rng.uniform(0.8, 1.2)
        for i in range(n):
            A[i][i] = base
            if i + 1 < n:
                A[i][i + 1] = 1e-3 if n > 6 else 1e-2
        for i in range(n):
            for j in range(n):
                A[i][j] += rng.uniform(-1e-4, 1e-4)
        return A

    if family == "rotation_block_mix" and n >= 2:
        # 2x2 rotation block creates complex eigen pair in real matrix
        theta = rng.uniform(0.1, 2.5)
        c, s = math.cos(theta), math.sin(theta)
        A[0][0], A[0][1], A[1][0], A[1][1] = c, -s, s, c
        # lightly couple with rest
        for i in range(2, n):
            A[0][i] *= 0.1
            A[1][i] *= 0.1
            A[i][0] *= 0.1
            A[i][1] *= 0.1
        return A

    return A


def _sample_matrix_difficulty(spec: DomainSpec, rng: random.Random) -> Tuple[str, int, str]:
    cfg = spec.params.get("difficulty", {})
    levels = cfg.get("levels", ["easy", "medium", "hard"])
    weights = cfg.get("weights", [0.4, 0.4, 0.2])

    total = sum(weights)
    x = rng.random() * total
    acc = 0.0
    level = levels[-1]
    for lv, w in zip(levels, weights):
        acc += w
        if x <= acc:
            level = lv
            break

    n_range = cfg.get("n_range", {}).get(level, [2, 10])
    families = cfg.get("families", {}).get(level, spec.params.get("families", ["diagonal_dominant"]))

    n = rng.randint(int(n_range[0]), int(n_range[1]))
    family = rng.choice(families)
    return level, n, family


def _matrix_instruction_pool() -> List[Dict[str, Any]]:
    return [
        {
            "variant": "core_speed",
            "capability_tags": ["numerical_linear_algebra", "latency_optimization"],
            "text": (
                "在 /app/eigen.py 中完善 find_dominant_eigenvalue_and_eigenvector。"
                "要求返回模长最大的特征值对应特征对；输入是实数 float64 方阵（最大 10x10），"
                "矩阵可非对称，因此结果可能为复数。请以 /app/eval.py 为评测参考，"
                "在保证 np.allclose(A @ eigenvec, eigenval * eigenvec) 通过的前提下，"
                "尽量降低多次测试下的 median 单次调用耗时。"
            ),
        },
        {
            "variant": "robust_complex",
            "capability_tags": ["complex_eigenpair_handling", "robustness"],
            "text": (
                "请补全 /app/eigen.py 的主特征对函数。这里的 dominant 指谱半径最大。"
                "评测会包含非对称样本与复特征值场景，维度不超过 10。"
                "你的实现必须满足 Av≈λv（按 np.allclose 判定），并且相对 /app/eval.py 的 numpy 参考路径"
                "在中位延迟上更有优势。"
            ),
        },
        {
            "variant": "engineering_tradeoff",
            "capability_tags": ["algorithm_selection", "engineering_delivery"],
            "text": (
                "目标：交付一个更快的 dominant eigenpair 求解入口（函数位置 /app/eigen.py）。"
                "输入固定为 np.float64 方阵，n<=10，且不保证对称。"
                "你可以自由选择实现策略（含第三方或其他语言加速），"
                "但必须保留 Python 入口并通过 /app/eval.py 的正确性与性能对比："
                "正确性检查为 np.allclose(A @ v, lambda * v)，性能按 median time/call 统计。"
            ),
        },
        {
            "variant": "hard_case_first",
            "capability_tags": ["hard_case_generalization", "non_symmetric_matrix"],
            "text": (
                "请优化 find_dominant_eigenvalue_and_eigenvector，使其在困难样本（非对称、近缺陷、"
                "可能出现复主特征值）下仍稳定返回主特征对。"
                "dominant 定义为特征值绝对值最大。输入规模最高 10x10 float64。"
                "我们会重复测量并比较 median 延迟，你需要在不破坏 allclose 正确性的前提下超过参考实现。"
            ),
        },
    ]


def matrix_sample(spec: DomainSpec, rng: random.Random) -> Dict[str, Any]:
    level, n, family = _sample_matrix_difficulty(spec, rng)
    A = _build_matrix(n, family, rng)
    lam_est, residual_est = estimate_dominant_eigenvalue(A)

    template = rng.choice(_matrix_instruction_pool())

    return {
        "instruction": template["text"],
        "context_files": ["/app/eigen.py", "/app/eval.py"],
        "inputs": {
            "A": A,
            "dtype": "float64",
            "shape": [n, n],
            "n_max": 10,
        },
        "reference_output": {
            "dominant_eigenvalue_estimate": lam_est,
            "residual_estimate": residual_est,
        },
        "constraints": {
            "dominant_by_magnitude": True,
            "allow_complex_eigenpair": True,
            "correctness_check": "np.allclose(A @ eigenvec, eigenval * eigenvec)",
            "performance_protocol": "median_time_per_call_vs_numpy_baseline",
        },
        "metrics": {
            "correctness": "allclose_pass_rate",
            "performance": "median_latency_speedup",
            "robustness": "hard_case_pass_rate",
        },
        "metadata": {
            "difficulty": level,
            "n": n,
            "family": family,
            "target_function": "find_dominant_eigenvalue_and_eigenvector",
            "instruction_variant": template["variant"],
            "capability_tags": template["capability_tags"],
        },
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
        "metrics": {"correctness": "exact_match", "performance": "records_per_second"},
    }


def graph_sample(spec: DomainSpec, rng: random.Random) -> Dict[str, Any]:
    n = rng.randint(spec.params.get("n_min", 20), spec.params.get("n_max", 80))
    d0, d1 = spec.params.get("density", [0.05, 0.2])
    density = rng.uniform(d0, d1)
    inf = 10**12
    dist = [[inf] * n for _ in range(n)]
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
    if best >= inf:
        best = None

    return {
        "instruction": "计算从0到n-1的最短路径代价",
        "inputs": {"n": n, "edges": edges, "src": 0, "dst": n - 1},
        "reference_output": {"shortest_cost": best},
        "constraints": {"non_negative_weight": True},
        "metrics": {"correctness": "exact_or_none", "performance": "median_latency"},
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
        "metrics": {"correctness": "exact_match", "performance": "rows_per_second"},
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
        "metrics": {"correctness": "exact_match", "reliability": "recovery_rate"},
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
        "metrics": {"correctness": "feasible_plan", "cost": "api_calls"},
    }


def build_sample(domain: DomainSpec, sid: int, rng: random.Random) -> Dict[str, Any]:
    fn = {
        "matrix": matrix_sample,
        "string_parsing": string_sample,
        "graph": graph_sample,
        "db_retrieval": db_sample,
        "system_concurrency": system_sample,
        "api_code": api_code_sample,
    }.get(domain.name)
    if fn is None:
        raise ValueError(f"unsupported domain: {domain.name}")
    s = fn(domain, rng)
    s["task_id"] = f"{domain.name}-{sid:06d}"
    s["domain"] = domain.name
    s.setdefault("difficulty", s.get("metadata", {}).get("difficulty", rng.choice(["easy", "medium", "hard"])))
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
    matrix_difficulty: Dict[str, int] = {}
    for x in items:
        stats[x["domain"]] = stats.get(x["domain"], 0) + 1
        if x["domain"] == "matrix":
            diff = x.get("difficulty", "unknown")
            matrix_difficulty[diff] = matrix_difficulty.get(diff, 0) + 1

    with (root / "domain_stats.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["domain", "count"])
        for k in sorted(stats):
            w.writerow([k, stats[k]])

    with (root / "matrix_difficulty_stats.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["difficulty", "count"])
        for k in sorted(matrix_difficulty):
            w.writerow([k, matrix_difficulty[k]])

    print(f"generated {total} samples -> {root}")


if __name__ == "__main__":
    main()
