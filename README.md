# Mini-R1: GRPO + 本地沙箱训练 Qwen2.5-Coder-1.5B

> 在 4070 Ti 16GB 消费级 GPU 上，用 GRPO 训练 1.5B 代码模型，HumanEval Pass@1 从 **51.83% → 69.51%（+17.68%）**。

> [!IMPORTANT]
> 这些分数来自训练使用的同一组 HumanEval 164 道任务（in-sample），用于比较训练配方，**不是 held-out 泛化成绩**。泛化能力仍需在独立代码基准上验证。

```
基座 Qwen2.5-Coder-1.5B-Instruct ──> GRPO 训练 ──> 微调后模型
        Pass@1 = 51.83%                              Pass@1 = 69.51% ⭐
```

---

## 📊 关键结果

| 版本 | Reward 设计 | Epoch | LR | Pass@1 | Δ baseline | 评注 |
|------|------------|-------|-----|--------|-----------|------|
| baseline | — | — | — | 51.83% | — | Qwen2.5-Coder-1.5B 原版 |
| v1 | 二元 (0/1) | 1 | 5e-6 | 51.22% | -0.61% | ❌ Reward 信号缺失 |
| v2 | **连续 (assert pass rate)** | 1 | 5e-6 | 58.54% | +6.71% | ✅ 修复 reward |
| **v3** | **连续** | **2** | **1e-5** | **69.51%** | **+17.68%** | ⭐ **最优解** |
| v4 | 连续 | 3 | 1e-5 | 62.20% | +10.37% | 📉 同任务性能回落 |

**4 轮迭代覆盖了完整的"失败 → 诊断 → 修复 → 优化 → 找到边界"过程。**
详细见 [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md)。

---

## 🎯 项目亮点

1. **本地 Python 子进程沙箱**
   替换 Open-R1 的付费 E2B / MorphCloud 云沙箱（需要 API key 和按使用付费）。
   纯 stdlib 实现，支持 stdin/stdout 测试 + HumanEval 函数调用测试两种模式；
   使用随机成功标记，避免只凭候选进程的退出码误判通过。

2. **AST-based 连续 Reward**
   解析 HumanEval `check()` 函数，在保留准备语句和控制流的同时为每个 `assert` 加入动态计数，
   按通过比例打分（0~1 连续）。
   把 GRPO 的 `reward_std=0` 比例从 **38% → 16%**，解决"同组全对/全错"导致的梯度信号缺失。

3. **小模型 + 消费级 GPU 适配**
   Qwen2.5-Coder-1.5B-Instruct + LoRA r=16 + bf16 + gradient checkpointing，
   在 RTX 4070 Ti SUPER 16GB 上 GRPO 训练全程显存 < 14 GB。

4. **完整消融实验**
   4 轮训练 (v1-v4) 验证每个改动的边际贡献，定量回答：
   - 二元 vs 连续 reward 差多少？(+6.71%)
   - 加 epoch 帮多少？(+11%)
   - 何时出现同任务性能回落？(第 3 epoch)

5. **诚实工程报告**
   v1 失败和 v4 性能回落都完整记录，并明确当前评测的数据复用边界。

---

## 🏗️ 整体架构

```
                       HumanEval (164 题)
                              │
                              ▼
               ┌──────────────────────────────┐
               │  data_prep.py                │
               │  转 chat 格式 + verification  │
               └──────────────────────────────┘
                              │
              ▼
   ┌──────────────────────────────────────────────┐
   │  Policy Model: Qwen + LoRA (训练)             │
   │  β = 0.0：不加载 Reference Model，无 KL 惩罚   │
   └──────────────────────────────────────────────┘
              │
              │ 生成 4 个 completion / 题
              ▼
   ┌─────────────────────────────────────────────┐
   │   reward_funcs.py                            │
   │                                              │
   │   ┌─────────────────────────────────────┐   │
   │   │ extract_code()  → 抠 ```python``` │   │
   │   └─────────────────────────────────────┘   │
   │                ↓                             │
   │   ┌─────────────────────────────────────┐   │
   │   │ local_sandbox.py                    │   │
   │   │   compute_humaneval_pass_rate()     │   │
   │   │   ├─ AST 改写 check() 中的 assert   │   │
   │   │   ├─ 单个子进程保留控制流并计数     │   │
   │   │   └─ 返回 pass_rate ∈ [0, 1]        │   │
   │   └─────────────────────────────────────┘   │
   │                ↓                             │
   │   reward = pass_rate + format_score          │
   └─────────────────────────────────────────────┘
              │
              ▼
   ┌──────────────────────────────────────────────┐
   │  GRPO 更新 (TRL 0.24)                         │
   │  组内归一化 → policy gradient（β=0，无 KL）    │
   └──────────────────────────────────────────────┘
```

---

## 🚀 快速开始

### 环境

```bash
conda create -n mini-r1 python=3.10 -y
conda activate mini-r1

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.56.1,<5.0.0" "trl>=0.24.0,<0.26.0" peft accelerate datasets bitsandbytes
```

### 一键复现

```bash
# 1. 评测基座 (10-20 min)
python src/evaluate.py --label baseline

# 2. 训练 v3 最优版：2 epochs (~4 hours on 4070 Ti)
python src/train.py --num_train_epochs 2 --output_dir outputs/grpo_humaneval_v3

# 3. 评测训练后模型
python src/evaluate.py \
  --lora_path outputs/grpo_humaneval_v3 \
  --label trained_v3
```

预期输出：
```
============================================================
Pass@1: 69.51%  (114/164)
============================================================
```

---

## 📂 项目结构

```
mini-r1/
├── src/
│   ├── local_sandbox.py    # 本地子进程沙箱 (I/O + HumanEval 双模式)
│   ├── reward_funcs.py     # 3 个 reward 函数
│   ├── data_prep.py        # HumanEval 数据加载
│   ├── train.py            # GRPO 训练脚本
│   ├── evaluate.py         # Pass@1 评测脚本
│   └── test_e2e.py         # 端到端测试
├── outputs/
│   ├── grpo_humaneval_v3/  # 最佳模型 (LoRA adapter)
│   └── eval/               # 4 个版本的评测 JSON
├── EXPERIMENT_LOG.md        # 完整实验记录
├── README.md
└── requirements.txt
```

---

## 🔬 核心实现

### 1. 本地沙箱 (`src/local_sandbox.py`)

替换 Open-R1 的付费云沙箱，纯 Python stdlib。真正的判定不只检查退出码：
包装脚本会先缓存判分所需的 `compile`、`exec`、`os.write` 和 `os._exit`，执行
`check(candidate)` 成功后才写出一次随机标记。下面是精简后的判定部分，完整包装逻辑见
[`src/local_sandbox.py`](src/local_sandbox.py)：

```python
success_marker = f"{_SUCCESS_MARKER_PREFIX}:{secrets.token_hex(16)}"

# full_script 仅在 check(candidate) 正常完成后写出 success_marker
result = subprocess.run(
    [sys.executable, "-c", full_script],
    capture_output=True,
    text=True,
    timeout=timeout,
)
if result.returncode != 0:
    return False

success_lines = [
    line for line in result.stdout.splitlines() if line == success_marker
]
return success_lines == [success_marker]
```

> 安全边界：子进程用于限制崩溃和超时，不是对抗恶意 Python 的安全边界。
> 执行不可信代码时仍应使用容器、低权限账户或专用沙箱。

### 2. 连续 Reward (核心创新)

二元 reward (`run_humaneval_test`) 在 GRPO 上信号不足。连续 reward 使用
`_instrument_humaneval_tests()` 改写 `check()` 中实际执行的 `assert`，在不展开循环、
不丢失准备语句的前提下动态累计通过数和总数。所有测试在单个子进程中执行：

```python
def compute_humaneval_pass_rate(
    code, test_code, entry_point, timeout=10
) -> float:
    instrumented_tests = _instrument_humaneval_tests(test_code)
    if instrumented_tests is None:
        return float(
            run_humaneval_test(code, test_code, entry_point, timeout)
        )

    pass_count_marker = f"{_PASS_COUNT_MARKER_PREFIX}:{secrets.token_hex(16)}="

    # full_script 执行改写后的 check()，并用随机 marker 回传 passed/total。
    result = subprocess.run(
        [sys.executable, "-c", full_script],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        return 0.0

    marker_lines = [
        line
        for line in result.stdout.splitlines()
        if line.startswith(pass_count_marker)
    ]
    if len(marker_lines) != 1:
        return 0.0

    try:
        passed_text, total_text = (
            marker_lines[0].removeprefix(pass_count_marker).split("/", 1)
        )
        passed, total = int(passed_text), int(total_text)
    except ValueError:
        return 0.0
    if total <= 0 or not 0 <= passed <= total:
        return 0.0
    return passed / total
```

上例省略了包装脚本的构造细节；展示的 marker 校验会拒绝缺失、重复、伪造或越界的计数，
不会信任候选代码自行打印的数字。

### 3. 训练配置 (v3 最优配方)

```python
# src/train.py
training_args = GRPOConfig(
    learning_rate=1e-5,            # ↑ from 5e-6
    num_train_epochs=2,            # ↑ from 1
    beta=0.0,                      # 不加载 reference model，无 KL 惩罚
    num_generations=4,             # GRPO group size
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_prompt_length=512,
    max_completion_length=512,
    bf16=True,
    gradient_checkpointing=True,
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[
        code_reward_humaneval_partial,  # 连续 reward (核心)
        format_reward,                   # 格式 reward
    ],
    args=training_args,
    train_dataset=ds,
    peft_config=LoraConfig(r=16, lora_alpha=32, ...),
)
```

为稳定复现 v3 配方，`beta=0.0` 被显式固定。TRL 0.24–0.25 在该配置下不加载参考模型，训练目标不包含 KL 惩罚。

---

## 📈 详细结果

### Pass@1 演进

```
%
70 ┤                                  ╭──── v3 (69.51%)
   │                                  │
65 ┤                                  │   ╲
   │                                  │    ╲
60 ┤                          ╭───────╯     ╲── v4 (62.20%)
   │                          │
55 ┤                          │  v2 (58.54%)
   │   baseline (51.83%)      │
50 ┤━━━━━━━━━━━━━━━━━━━━━━━━━━╯  v1 (51.22%)
   └────────────────────────────────────────────────
       v0           v1          v2          v3       v4
```

### 训练信号 (reward_std=0 比例)

```
%
40 ┤  v1 (38%)  ← 二元 reward, 信号崩溃
35 ┤
30 ┤              v3 (23%) ←──┐
25 ┤                          │
20 ┤    v2 (16%) ──────────────┴── v4 (31%) "好学生塌缩"
15 ┤
   └─────────────────────────────────
      v1        v2        v3       v4
```

---

## 💡 关键洞察

### 1. **Reward 设计 > 训练时长**
v1→v2 仅改 reward 函数，Pass@1 直接 +6.71%。同样 1 epoch 同样 LR。

### 2. **GRPO 在小模型上对 reward_std 极度敏感**
v1 的 `reward_std=0` 比例 38% → 训练梯度大部分时候为 0 → 学习近乎随机。
v2 的连续 reward 把这个比例降到 16% → 立刻见效。

### 3. **小数据 RL 的同任务评测存在性能峰值**
164 样本 × 2 epoch (v3) 在当前实验中最佳。再加 1 个 epoch (v4) 后，
训练 reward 仍在上涨，但同任务 Pass@1 回落 7.31%。这提示策略退化或过度优化，
不能单凭本实验断言 held-out 泛化过拟合。

### 4. **训练 reward ≠ 评测 Pass@1**
v4 训练 reward (1.053) > v3 (1.011)，但 Pass@1 反降。
当前结果不是 holdout；泛化结论必须由独立代码基准支持。

---

## 🆚 vs 其他项目

| 项目 | Star | 基座 | GPU 需求 | 沙箱方式 |
|------|------|------|---------|---------|
| Open-R1 (HuggingFace) | 26k | Qwen2.5 全系 | 多 GPU | E2B / Morph 付费 |
| DeepCoder (Together AI) | 356 | Qwen-14B | 32×H100 | verl |
| CURE (NeurIPS 2025) | 165 | Qwen-7B/14B | 多 A100 | 内置 |
| **本项目** | — | **Qwen-1.5B** | **1×4070Ti 16GB** | **本地 subprocess** |

定位：**消费级 GPU 上能跑通 + 消融完整的 GRPO 教学/原型项目**。

---

## 🛠️ 技术栈

- **PyTorch 2.x** + **CUDA 12.1**
- **HuggingFace Transformers** 4.46+
- **TRL** 0.24 (GRPOTrainer)
- **PEFT** (LoRA)
- **Datasets** (HumanEval 加载)

---

## 🙏 致谢

- [Open-R1 (HuggingFace)](https://github.com/huggingface/open-r1) — 框架结构、`extract_code` 函数借鉴
- [TinyZero (Jiayi-Pan)](https://github.com/Jiayi-Pan/TinyZero) — R1-Zero 复现思路启发
- [CURE (Gen-Verse, NeurIPS 2025)](https://github.com/Gen-Verse/CURE) — 代码 + 测试协同 reward 设计参考
- [DeepSeek-R1 论文](https://arxiv.org/abs/2501.12948) — GRPO 算法
- HumanEval 数据集 (OpenAI)

---

## 📝 简历话术模板

> 基于 HuggingFace Open-R1 框架，在 RTX 4070 Ti 16GB 消费级 GPU 上完成 Qwen2.5-Coder-1.5B 的 GRPO 训练。
>
> **三个核心贡献**：
> 1. 用本地 Python subprocess 沙箱替换付费 E2B 云沙箱，支持 I/O + 函数调用双模式测试。
> 2. 通过 AST 解析 HumanEval `check()` 函数，将二元 reward 改为按 assert 通过率的连续 reward (0~1)，把 GRPO 训练中 `reward_std=0` 的比例从 38% 降到 16%，解决了小模型 GRPO 的梯度信号缺失问题。
> 3. 完成 4 轮消融实验 (v1-v4)，同任务 HumanEval Pass@1 从 51.83% → **69.51% (+17.68%)**；第 3 epoch 回落 7.31%，并据此识别出独立泛化评测的必要性。

---

## 📜 License

MIT
