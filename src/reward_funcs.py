"""
GRPO 训练用的 reward 函数

接口约定 (来自 TRL GRPOTrainer 文档):
    def reward_func(completions, **kwargs) -> list[float]:
        ...

completions 格式 (chat 形式):
    [
        [{"role": "assistant", "content": "模型的回答1"}],
        [{"role": "assistant", "content": "模型的回答2"}],
        ...
    ]

verification_info 格式 (从数据集通过 kwargs 传入):
    [
        {"language": "python", "test_cases": [{"input": "...", "output": "..."}, ...]},
        ...
    ]

运行方式:
    python src/reward_funcs.py   # 自测
"""

import re
import sys
from pathlib import Path
from typing import List

# 把 src/ 加入 import 路径，这样能 import 同目录的 local_sandbox
sys.path.insert(0, str(Path(__file__).parent))
from local_sandbox import compute_pass_rate, run_humaneval_test, compute_humaneval_pass_rate


_FORMAT_PATTERN = re.compile(
    r"\s*<reasoning>(?:(?!</?(?:reasoning|answer)(?=[\s/>])).)*</reasoning>"
    r"\s*<answer>(?:(?!</?(?:reasoning|answer)(?=[\s/>])).)*</answer>\s*",
    re.DOTALL,
)


def _pair_completions_with_metadata(completions, verification_info):
    """Pair aligned reward inputs without silently dropping samples."""
    completion_count = len(completions)
    metadata_count = len(verification_info)
    if completion_count != metadata_count:
        raise ValueError(
            "completions and verification_info must have the same length "
            f"(got {completion_count} and {metadata_count})"
        )
    return zip(completions, verification_info)


# ========== Reward 1: 代码正确性 ==========

def extract_code(completion_text: str) -> str:
    """从模型输出里抠 ```python ... ``` 代码块

    和 Open-R1 的 extract_code 完全一样:
      - 正则匹配 ```python\n...```
      - 取最后一个匹配 (模型可能写多个代码块)
    """
    pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion_text)
    return matches[-1] if matches else ""


def code_reward(completions, **kwargs) -> List[float]:
    """主奖励: 代码在本地沙箱跑出的 pass rate

    返回每个 completion 的 pass rate (0 ~ 1)
    - 抠不出代码 → 0.0
    - 代码报错 → 0.0
    - 全部测试通过 → 1.0
    - 部分通过 → 0.x
    """
    verification_info = kwargs["verification_info"]
    pairs = _pair_completions_with_metadata(completions, verification_info)
    rewards = []

    for completion, info in pairs:
        text = completion[-1]["content"]      # 取 assistant 回答内容
        code = extract_code(text)

        if not code:                           # 没代码块
            rewards.append(0.0)
            continue

        rate = compute_pass_rate(code, info["test_cases"])
        rewards.append(rate)

    return rewards


# ========== Reward 1b: HumanEval 版本的代码正确性 ==========

def code_reward_humaneval(completions, **kwargs) -> List[float]:
    """HumanEval 版 code_reward (函数调用测试, 二元 0/1)

    和 code_reward 的区别:
      - code_reward:           stdin/stdout 测试, 返回 pass rate (0~1)
      - code_reward_humaneval: check(函数) 测试, 返回二元分数 (0 或 1)

    verification_info 格式:
        {"test_code": "def check(candidate): ...", "entry_point": "函数名"}
    """
    verification_info = kwargs["verification_info"]
    pairs = _pair_completions_with_metadata(completions, verification_info)
    rewards = []

    for completion, info in pairs:
        text = completion[-1]["content"]
        code = extract_code(text)

        if not code:
            rewards.append(0.0)
            continue

        passed = run_humaneval_test(
            code=code,
            test_code=info["test_code"],
            entry_point=info["entry_point"],
        )
        rewards.append(1.0 if passed else 0.0)

    return rewards


# ========== Reward 1c: HumanEval 连续版 (核心改动) ==========

def code_reward_humaneval_partial(completions, **kwargs) -> List[float]:
    """HumanEval 连续 reward — 按 assert 通过比例打分 (0~1)

    和 code_reward_humaneval 的区别:
      - code_reward_humaneval:         全过→1.0, 任一失败→0.0  (二元)
      - code_reward_humaneval_partial: 7 个过 4 个 → 0.57       (连续)

    连续信号:
      - 缓解 reward_std=0 (同组全 0 或全 1 的情况)
      - 给"接近对"的答案部分分, 梯度信号更丰富
    """
    verification_info = kwargs["verification_info"]
    pairs = _pair_completions_with_metadata(completions, verification_info)
    rewards = []

    for completion, info in pairs:
        text = completion[-1]["content"]
        code = extract_code(text)

        if not code:
            rewards.append(0.0)
            continue

        pass_rate = compute_humaneval_pass_rate(
            code=code,
            test_code=info["test_code"],
            entry_point=info["entry_point"],
        )
        rewards.append(pass_rate)

    return rewards


# ========== Reward 2: 输出格式 ==========

def format_reward(completions, **kwargs) -> List[float]:
    """格式奖励: 鼓励 <reasoning>...</reasoning><answer>...</answer> 结构

    - 整个回答符合单一完整格式 (首尾可有空白) → 0.5
    - 额外文本、重复区块或嵌套结构标签 → 0.0
    - 没有格式 → 0.0

    这个 reward 分值故意比 code_reward 小 (max 0.5 vs max 1.0),
    避免模型为了格式分而放弃做对题目。
    """
    rewards = []

    for completion in completions:
        text = completion[-1]["content"]
        if _FORMAT_PATTERN.fullmatch(text):
            rewards.append(0.5)
        else:
            rewards.append(0.0)

    return rewards


# ========== 自测 ==========
if __name__ == "__main__":

    # 模拟 TRL 传进来的 4 个 completions
    fake_completions = [
        # 1. 格式正确 + 代码正确
        [{"role": "assistant", "content": """<reasoning>
Read two numbers from stdin and print their sum.
</reasoning>
<answer>
```python
import sys
a, b = map(int, sys.stdin.read().split())
print(a + b)
```
</answer>"""}],

        # 2. 格式正确但代码错误 (用了减法)
        [{"role": "assistant", "content": """<reasoning>
Subtract them.
</reasoning>
<answer>
```python
import sys
a, b = map(int, sys.stdin.read().split())
print(a - b)
```
</answer>"""}],

        # 3. 没格式但代码正确
        [{"role": "assistant", "content": """```python
import sys
a, b = map(int, sys.stdin.read().split())
print(a + b)
```"""}],

        # 4. 什么都没有
        [{"role": "assistant", "content": "I don't know how to solve this."}],
    ]

    # 每个 completion 对应一组测试用例 (GRPO 里一组 completion 对应同一道题,
    # 但这里为了简单,假设 4 个都是 "两数之和" 这道题)
    fake_verification_info = [
        {
            "language": "python",
            "test_cases": [
                {"input": "1 2", "output": "3"},
                {"input": "10 20", "output": "30"},
            ]
        }
    ] * 4   # 复制 4 份

    print("=== 测试 code_reward ===")
    code_scores = code_reward(fake_completions, verification_info=fake_verification_info)
    print(f"分数: {[f'{s:.2f}' for s in code_scores]}")
    print(f"期望: [1.00, 0.00, 1.00, 0.00]  (正确、错误、正确、无代码)")

    print("\n=== 测试 format_reward ===")
    format_scores = format_reward(fake_completions)
    print(f"分数: {format_scores}")
    print(f"期望: [0.5, 0.5, 0.0, 0.0]  (有格式、有格式、无格式、无格式)")

    print("\n=== 组合 (code + format) ==")
    total = [c + f for c, f in zip(code_scores, format_scores)]
    print(f"总分: {[f'{s:.2f}' for s in total]}")
    print(f"期望: [1.50, 0.50, 1.00, 0.00]")
