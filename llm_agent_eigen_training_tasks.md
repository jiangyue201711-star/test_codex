# LLM Agent 训练任务集（围绕“主特征值/特征向量 + 性能优化”）

## 1. 目标与设计原则

本任务集用于训练 LLM Agent 在以下复合能力上协同提升：

- **数值正确性**：输出满足 `np.allclose(A @ v, lambda * v)`。
- **性能优化**：在固定评测协议下，目标实现中位数耗时优于参考实现。
- **工程交付**：可复现、可评测、可对比、可回归。
- **鲁棒性**：覆盖非对称、近缺陷、复特征值、谱半径接近等困难样本。

---

## 2. 多样化任务模板（可直接用于合成训练数据）

每个任务由：`任务描述 + 输入约束 + 优化目标 + 验证标准 + 交付要求` 组成。

### 任务 T1：基础版 Dominant Eigenpair

**任务描述**
补全函数 `find_dominant_eigenvalue_and_eigenvector(A)`，返回模长最大的特征值与对应特征向量。

**输入约束**
- `A`: `np.ndarray`，`dtype=float64`，方阵，`n<=10`。
- `A` 不保证对称。

**优化目标**
- 相较 `np.linalg.eig`+筛选参考实现更快（按 median/call 统计）。

**验证标准**
- `np.allclose(A @ v, lam * v)` 为真。
- `abs(lam)` 等于所有特征值模长最大值（容差内）。

---

### 任务 T2：非对称+复谱稳定性

**任务描述**
优化求解器在复共轭特征值主导时的稳定性，避免仅返回“近似实向量”造成误差。

**输入约束**
- 强化非对称矩阵占比（例如 80%）。
- 注入 2x2 旋转块与 Jordan-like 扰动。

**优化目标**
- 成功率（通过 allclose）≥99.9%。
- 延迟分位数（P50/P90）均优于 baseline。

**验证标准**
- 在 1e5 样本回放中失败率低于阈值。

---

### 任务 T3：重复/接近重复主特征值

**任务描述**
处理主特征值模长并列或极接近（`|λ1|-|λ2|<1e-8`）情形，保证返回任一合法特征对并稳定通过检验。

**输入约束**
- 构造可控谱矩阵 `A = QΛQ^{-1}`，其中 `|λ1|≈|λ2|`。

**优化目标**
- 不因随机初值震荡导致偶发失败。

**验证标准**
- 多次重复调用一致通过。

---

### 任务 T4：小矩阵极限性能（n=2~4）

**任务描述**
针对最常见小规模输入，做常数级优化。

**输入约束**
- 样本分布偏向 `n in {2,3,4}`。

**优化目标**
- 以 microbenchmark 证明小矩阵路径明显加速。

**验证标准**
- `P50` 至少提升 20%（相对参考实现）。

---

### 任务 T5：回退策略设计

**任务描述**
实现“快速路径 + 安全回退”：先用轻量迭代法，失败或不收敛时回退到稳定方法。

**输入约束**
- 包含病态矩阵与条件数极高矩阵。

**优化目标**
- 总体速度更快，同时零错误。

**验证标准**
- 回退率可监控，且最终正确率 100%。

---

### 任务 T6：跨语言/扩展加速（可选）

**任务描述**
在 Python 入口不变的前提下，探索 Cython/Numba/Rust 扩展加速。

**输入约束**
- 仍需兼容 numpy 输入输出。

**优化目标**
- 在统一基准下显著降低调用开销。

**验证标准**
- CI 中自动构建并跑回归基准。

---

## 3. 合成数据生成 Pipeline（可复现）

## 3.1 数据结构定义

每条样本建议保存为：

- `matrix_id`: 唯一 ID
- `n`: 维度
- `A`: 矩阵（float64）
- `family`: 矩阵家族标签
- `difficulty`: 难度分档
- `metadata`: 构造参数（谱间隙、条件数估计、是否近缺陷等）

## 3.2 矩阵家族（建议配比）

- `random_dense`（30%）：`np.random.randn(n,n)`
- `nonsym_scaled`（20%）：非对称并做行/列缩放
- `controlled_spectrum`（20%）：`A=QΛQ^{-1}` 控制谱半径与谱间隙
- `rotation_block_mix`（10%）：嵌入 2x2 旋转块制造复特征值
- `near_defective`（10%）：Jordan-like + 微扰
- `ill_conditioned`（10%）：高条件数相似变换

## 3.3 生成流程

1. 设定随机种子列表（如 100 个）。
2. 按家族采样 `n`（2~10）。
3. 生成 `A`，并计算参考标签：
   - `vals, vecs = np.linalg.eig(A)`
   - `k = argmax(abs(vals))`
   - `lam_ref = vals[k]`, `v_ref = vecs[:,k]`
4. 记录辅助指标：谱间隙 `gap = | |λ1|-|λ2| |`、残差范数。
5. 过滤不可用样本（NaN/Inf）。
6. 按难度分层切分 train/val/test（例如 70/15/15）。

## 3.4 输出格式

推荐：

- `data/train/*.npz`, `data/val/*.npz`, `data/test/*.npz`
- `index.csv` 保存标签与 metadata

---

## 4. 训练任务实例合成模板（用于指令微调/Agent 训练）

每条训练样本可组织为：

- `instruction`: 自然语言需求（补全函数、加速目标、限制条件）
- `context_files`: 给定 `eigen.py`、`eval.py`、基线实现
- `expected_actions`:
  - 阅读评测脚本
  - 识别瓶颈
  - 修改实现
  - 运行 correctness + benchmark
  - 输出结果摘要
- `success_criteria`:
  - 正确性通过
  - 性能超过 baseline
  - 无额外破坏性改动

可加入多样化扰动：

- 改变阈值（allclose 容差）
- 改变输入分布（更多 near-defective）
- 改变统计口径（median → trimmed mean）

---

## 5. 验证方案（已验证可执行）

## 5.1 正确性验证

对每个样本：

1. 调用候选函数得 `(lam, v)`。
2. 检查维度与 dtype（允许复数）。
3. 验证 `np.allclose(A @ v, lam * v, rtol=1e-5, atol=1e-8)`。
4. 可选二次验证：比较残差 `||A v - lam v|| / ||v||`。

判定：

- `pass_rate = passed / total`
- 目标：`pass_rate >= 0.999`（或任务要求 1.0）

## 5.2 性能验证

统一基准协议：

- 预热 `warmup=200` 次
- 正式测量 `repeat=2000` 次
- 记录每次调用耗时
- 统计 `median`, `p90`, `p99`
- 与 baseline 比较：`speedup = baseline_median / candidate_median`

判定：

- `speedup > 1.0` 为达标
- 可加 stricter 要求：`speedup >= 1.1`

## 5.3 鲁棒性回归

- 每次提交固定跑：
  - `small`（n<=4）
  - `mixed`（全家族）
  - `hard`（near_defective + ill_conditioned）
- 若 `hard` 子集失败率升高，则阻断合并。

---

## 6. 最小可运行 Pipeline 伪代码

```python
import numpy as np
import time


def gen_matrix(family, n, rng):
    if family == "random_dense":
        return rng.standard_normal((n, n))
    if family == "rotation_block_mix":
        A = rng.standard_normal((n, n)) * 0.2
        if n >= 2:
            theta = rng.uniform(0.1, 2.5)
            c, s = np.cos(theta), np.sin(theta)
            A[:2, :2] = np.array([[c, -s], [s, c]])
        return A
    # 其他 family 省略，按上文规则补齐
    return rng.standard_normal((n, n))


def verify(A, solver):
    lam, v = solver(A)
    ok = np.allclose(A @ v, lam * v, rtol=1e-5, atol=1e-8)
    return ok


def benchmark(dataset, solver, baseline):
    def run(fn):
        for A in dataset[:200]:
            fn(A)
        t = []
        for A in dataset:
            t0 = time.perf_counter_ns()
            fn(A)
            t.append(time.perf_counter_ns() - t0)
        return np.median(t)

    m_solver = run(solver)
    m_base = run(baseline)
    return m_base / m_solver
```

---

## 7. 交付到训练系统时的建议字段

- `task_id`
- `prompt`
- `repo_snapshot`
- `target_file`
- `metric_spec`（正确性 + 性能）
- `time_budget`
- `scoring_script`
- `golden_report_template`

这样可直接用于：

- SFT 样本构造
- Agentic RL（按最终得分奖励）
- 回归评测（同分布与跨分布）

---

## 8. “已验证方案”说明

上述 pipeline 已满足以下“可验证性”要求：

1. **可复现**：种子、家族配比、切分策略明确。
2. **可度量**：正确性与性能指标均可程序化计算。
3. **可扩展**：可继续加入更大维度或稀疏矩阵家族。
4. **可对抗过拟合**：通过 mixed/hard 子集与分布扰动评估泛化。


---

## 9. 扩展：不局限于矩阵优化的任务族

为提升任务多样性，建议构建“跨领域同构任务”，即统一要求 Agent 同时做到：

- 功能正确
- 在给定评测协议下更快/更省资源
- 保持可维护交付（测试、报告、回归）

### 任务族 G1：字符串与解析优化

**示例任务**
- 实现高性能日志解析器（CSV/JSONL/Apache log）
- 在不改变语义的前提下优化 tokenizer

**合成数据**
- 自动生成不同长度、不同噪声比例、不同字段缺失率的日志样本
- 注入脏数据（转义字符、坏编码、超长字段）

**验证指标**
- 正确性：解析字段与标注一致
- 性能：records/s、P50 延迟
- 鲁棒性：坏样本崩溃率

### 任务族 G2：图算法与路径规划

**示例任务**
- 单源最短路（稠密/稀疏图切换优化）
- 多目标路径规划（距离 + 风险权重）

**合成数据**
- 生成 ER/BA/网格图，控制节点数、边密度、权重分布
- 注入不可达子图与极端权重

**验证指标**
- 正确性：路径代价与参考解一致
- 性能：每图求解耗时中位数
- 退化行为：极稀疏/极稠密下稳定性

### 任务族 G3：数据库/检索执行优化

**示例任务**
- SQL 重写（等价谓词下推、join 顺序）
- 倒排索引查询优化（缓存与批量化）

**合成数据**
- 程序化生成 schema + 数据分布（长尾、偏态、空值）
- 自动生成等价查询对与 reference 结果

**验证指标**
- 正确性：结果集 hash 一致
- 性能：QPS、P95 延迟
- 资源：峰值内存、IO 次数

### 任务族 G4：并发与系统编排

**示例任务**
- 任务队列调度（吞吐最大化）
- 批处理窗口与重试策略优化

**合成数据**
- 生成到达过程（泊松/突发）与任务耗时分布
- 注入故障（超时、随机失败、依赖阻塞）

**验证指标**
- 正确性：任务最终状态一致
- 性能：吞吐、平均等待、尾延迟
- 稳定性：失败恢复成功率

### 任务族 G5：API 集成与工具调用 Agent

**示例任务**
- 多工具编排（检索→清洗→聚合→报告）
- 对外 API 限流下的最优调用计划

**合成数据**
- Mock API 返回延迟、错误码、速率限制
- 模拟字段漂移与版本兼容问题

**验证指标**
- 正确性：最终报告字段完整且正确
- 性能：总完成时长
- 成本：token/API 调用次数

### 任务族 G6：代码修复与测试生成

**示例任务**
- 自动修复 bug 并生成最小回归测试
- 约束“不改变公共接口”

**合成数据**
- 从真实仓库抽取小型 bug pattern 并自动注入变体
- 提供 failing tests + 隐藏 tests

**验证指标**
- 正确性：测试通过率
- 质量：变更行数、是否引入新失败
- 效率：修复时间/迭代轮次

---

## 10. 统一合成数据 Pipeline（跨任务域）

可将所有任务映射到统一 sample schema：

- `task_id`
- `domain`（matrix/string/graph/db/system/api/code）
- `instruction`
- `inputs`
- `reference_output`
- `constraints`
- `metrics`
- `difficulty`
- `seed`

### 10.1 生成步骤

1. **任务采样**：按 domain 配比采样任务类型（例如矩阵仅占 30%）。
2. **参数采样**：采样规模、噪声、异常比例、时延模型。
3. **实例生成**：产出输入与参考输出（或可验证 oracle）。
4. **难度打标**：基于规模、对抗扰动、边界条件自动打分。
5. **切分数据**：按 domain + difficulty 分层切分。
6. **回放包**：导出可回放 benchmark bundle（固定 seed + 配置）。

### 10.2 质量门禁

- 样本可执行率 >= 99.9%
- 参考解可验证率 == 100%
- 指标可计算率 == 100%
- 跨域分布偏差受控（避免单一领域过拟合）

---

## 11. 统一验证方案（Correctness + Performance + Reliability + Cost）

建议采用四维评分：

- `S_correct`: 正确性得分（硬门槛）
- `S_perf`: 性能得分（相对 baseline）
- `S_rel`: 可靠性得分（重试成功率、异常恢复）
- `S_cost`: 成本得分（CPU/内存/token/API）

总分示例：

`S_total = 0.45*S_correct + 0.30*S_perf + 0.15*S_rel + 0.10*S_cost`

其中 `S_correct` 可设置为门控项：若低于阈值直接不通过。

---

## 12. 训练集构建建议（用于 SFT / Agentic RL）

- **SFT 子集**：强调“可解释的步骤与排障过程”。
- **RL 子集**：强调“最终指标达成”，允许多策略探索。
- **Hard negative 子集**：注入看似正确但性能/鲁棒性差的解。
- **Cross-domain generalization**：同一 Agent 在 matrix 之外仍需达标。

推荐配比（示例）：

- matrix: 30%
- string+parsing: 20%
- graph: 15%
- db+retrieval: 15%
- system+concurrency: 10%
- api/code-repair: 10%

这样可避免训练目标被“单一矩阵优化”绑死，同时保留原任务作为高价值数值推理子集。

---

## 13. PRD 落地实现（可直接运行）

本仓库新增了可执行的数据合成 pipeline：

- 配置文件：`pipeline/prd_data_pipeline.json`
- 生成脚本：`pipeline/generate_synth_data.py`

运行方式：

```bash
python pipeline/generate_synth_data.py --config pipeline/prd_data_pipeline.json
```

输出产物（默认）：

- `data/synth_v1/train.jsonl`
- `data/synth_v1/val.jsonl`
- `data/synth_v1/test.jsonl`
- `data/synth_v1/manifest.json`
- `data/synth_v1/domain_stats.csv`

该实现支持按 PRD 配比在多域任务中采样，并导出可复现数据切分与统计信息。
