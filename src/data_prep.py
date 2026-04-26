"""
HumanEval 数据准备 —— 把 164 道题转成 GRPO 训练需要的格式

HumanEval 原始格式:
    {
        "task_id": "HumanEval/0",
        "prompt": "def add(a, b):\\n    '''...'''\\n    ",    # 函数签名+docstring
        "canonical_solution": "    return a + b",            # 官方答案
        "test": "def check(candidate):\\n    assert...",      # 测试函数
        "entry_point": "add",                                 # 函数名
    }

我们要的 GRPO 格式:
    {
        "prompt": [                        # chat 消息, 喂给模型
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Complete this Python function: ..."},
        ],
        "verification_info": {             # 测试用, 传给 reward 函数
            "test_code": "...",            # check() 函数代码
            "entry_point": "add",          # 函数名
        },
    }

运行方式:
    python src/data_prep.py   # 下载 + 预览
"""

from datasets import load_dataset


# 系统提示: 告诉模型怎么回答
SYSTEM_PROMPT = """You are a helpful Python programmer. Respond in the following format:
<reasoning>
[Step-by-step thinking about the problem]
</reasoning>
<answer>
```python
[Your complete Python function here]
```
</answer>"""


def load_humaneval(split: str = "test"):
    """加载 HumanEval 数据集并转成 GRPO 格式

    Args:
        split: HumanEval 只有 'test' split (164 道题, 都当训练用)

    Returns:
        datasets.Dataset, 每个样本包含:
            - prompt: chat 格式的输入
            - verification_info: reward 函数用的测试信息
            - task_id: 原题 ID (方便 debug)
    """
    ds = load_dataset("openai/openai_humaneval", split=split)

    def transform(example):
        # 构造用户问题: 函数签名 + 要求补全
        user_content = (
            f"Complete the following Python function:\n\n"
            f"```python\n{example['prompt']}```"
        )

        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "verification_info": {
                "test_code": example["test"],
                "entry_point": example["entry_point"],
            },
            "task_id": example["task_id"],
        }

    ds = ds.map(transform)
    return ds


if __name__ == "__main__":
    print("=" * 60)
    print("下载并转换 HumanEval...")
    print("=" * 60)

    ds = load_humaneval("test")
    print(f"\n数据集大小: {len(ds)} 道题")

    # --- 看第一道题长啥样 ---
    print("\n" + "=" * 60)
    print("第 1 道题预览 (HumanEval/0):")
    print("=" * 60)

    sample = ds[0]
    print(f"\ntask_id: {sample['task_id']}")

    print(f"\nprompt (user):")
    print(sample["prompt"][1]["content"])

    print(f"\nverification_info:")
    print(f"  entry_point: {sample['verification_info']['entry_point']}")
    print(f"  test_code (前 5 行):")
    for line in sample["verification_info"]["test_code"].split("\n")[:5]:
        print(f"    {line}")
    print(f"    ...")

    # --- 统计信息 ---
    print("\n" + "=" * 60)
    print("数据集统计:")
    print("=" * 60)

    prompt_lens = [len(s["prompt"][1]["content"]) for s in ds]
    print(f"平均 prompt 长度: {sum(prompt_lens) / len(prompt_lens):.0f} 字符")
    print(f"最短 prompt: {min(prompt_lens)} 字符")
    print(f"最长 prompt: {max(prompt_lens)} 字符")

    print("\n[OK] 数据准备完成!")
