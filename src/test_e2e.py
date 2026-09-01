"""
端到端测试: 拿 HumanEval 的官方答案喂给 reward 函数, 应该全部得满分

这个测试验证:
  1. 数据加载对了 (data_prep.py)
  2. 沙箱跑对了 (local_sandbox.py)
  3. reward 函数接口对了 (reward_funcs.py)

运行方式:
    python src/test_e2e.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from data_prep import load_humaneval
from reward_funcs import code_reward_humaneval_partial, format_reward


def fake_model_output(canonical_solution: str, prompt: str, with_format: bool = True) -> str:
    """模拟模型输出:
    - prompt: HumanEval 给的函数签名+docstring
    - canonical_solution: HumanEval 给的标准答案 (只是函数体, 没签名)
    - 需要把 prompt 和 solution 拼起来才是完整代码
    """
    full_code = prompt + canonical_solution

    if with_format:
        return f"""<reasoning>
I'll solve this step by step.
</reasoning>
<answer>
```python
{full_code}
```
</answer>"""
    else:
        return f"```python\n{full_code}\n```"


def main() -> int:
    failed_checks = 0
    print("加载 HumanEval (前 3 道题)...")
    ds = load_humaneval("test").select(range(3))

    # 需要原始 canonical_solution 和 prompt, 重新加载一份原始数据
    from datasets import load_dataset
    raw = load_dataset("openai/openai_humaneval", split="test").select(range(3))

    # 构造 completions (模拟模型输出 = 官方答案)
    fake_completions = [
        [{"role": "assistant", "content": fake_model_output(
            canonical_solution=raw[i]["canonical_solution"],
            prompt=raw[i]["prompt"],
            with_format=True,
        )}]
        for i in range(3)
    ]

    verification_info = [ds[i]["verification_info"] for i in range(3)]

    print("\n" + "=" * 60)
    print("测试 1: code_reward_humaneval_partial (官方答案应该全 1.0)")
    print("=" * 60)
    scores = code_reward_humaneval_partial(
        fake_completions,
        verification_info=verification_info,
    )
    expected_scores = [1.0] * len(fake_completions)
    for i, (s, expected) in enumerate(zip(scores, expected_scores)):
        passed = s == expected
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} Task {ds[i]['task_id']}: reward = {s:.2f}")
    if scores != expected_scores:
        print(f"  [FAIL] 期望 rewards={expected_scores}, 实际 {scores}")
        failed_checks += 1

    print("\n" + "=" * 60)
    print("测试 2: format_reward (都按格式写的, 应该全 0.5)")
    print("=" * 60)
    scores = format_reward(fake_completions)
    expected_scores = [0.5] * len(fake_completions)
    for i, (s, expected) in enumerate(zip(scores, expected_scores)):
        passed = s == expected
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} Task {ds[i]['task_id']}: reward = {s}")
    if scores != expected_scores:
        print(f"  [FAIL] 期望 rewards={expected_scores}, 实际 {scores}")
        failed_checks += 1

    print("\n" + "=" * 60)
    print("测试 3: 故意写错的代码 (必然报错的负对照)")
    print("=" * 60)
    wrong_completion = [[{"role": "assistant", "content": """<reasoning>test</reasoning>
<answer>
```python
def has_close_elements(numbers, threshold):
    raise RuntimeError("negative control")
```
</answer>"""}]]
    scores = code_reward_humaneval_partial(
        wrong_completion,
        verification_info=[verification_info[0]],
    )
    passed = scores == [0.0]
    print(f"  期望 [0.0], 实际 {scores}")
    print(f"  {'[PASS]' if passed else '[FAIL]'}")
    if not passed:
        failed_checks += 1

    if failed_checks:
        print(f"\n[FAIL] 端到端测试失败 ({failed_checks} 项)")
        return 1

    print("\n[OK] 端到端测试完成!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
