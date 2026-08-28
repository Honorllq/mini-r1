"""
HumanEval 评测脚本 — 对比训练前 vs 训练后的 Pass@1

用法:
    # 评测基座模型 (训练前)
    python src/evaluate.py --label baseline

    # 评测训练后的模型
    python src/evaluate.py --lora_path outputs/grpo_humaneval --label trained

    # 也可以只评测前 N 题加快速度
    python src/evaluate.py --num_samples 50

输出:
    outputs/eval/eval_<label>.json   - 详细结果
    控制台打印 Pass@1 总分
"""

from __future__ import annotations   # Python 3.9 兼容: 让 str | None 这种语法可用

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from data_prep import load_humaneval
from reward_funcs import extract_code
from local_sandbox import run_humaneval_test


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer for CLI sample counts."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer greater than 0") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def evaluate(
    model_name: str,
    lora_path: str | None = None,
    num_samples: int = 164,
    max_new_tokens: int = 512,
    label: str = "model",
    output_dir: str = "outputs/eval",
) -> dict:
    """评测模型在 HumanEval 上的 Pass@1

    Args:
        model_name: HF 模型名 (基座)
        lora_path: 如果有 LoRA, 给路径
        num_samples: 评多少题, 必须是大于 0 的整数, 最多 164
        max_new_tokens: 生成最大 token 数
        label: 标签, 用于保存结果文件名
        output_dir: 结果保存目录
    """
    if isinstance(num_samples, bool) or not isinstance(num_samples, int):
        raise TypeError("num_samples must be an integer")
    if num_samples <= 0:
        raise ValueError("num_samples must be greater than 0")

    # === 加载模型 ===
    print(f"\n[1/3] 加载模型: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if lora_path:
        from peft import PeftModel
        print(f"  套上 LoRA: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # === 加载数据 ===
    print(f"\n[2/3] 加载 HumanEval (前 {num_samples} 题)")
    ds = load_humaneval("test").select(range(min(num_samples, 164)))

    # === 推理 + 评测 ===
    print(f"\n[3/3] 开始评测 {len(ds)} 道题")
    results = []
    passed_count = 0
    no_code_count = 0

    for sample in tqdm(ds):
        # 构造输入
        prompt_text = tokenizer.apply_chat_template(
            sample["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        # 贪心生成 (评测用确定性, 不采样)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # 只取生成部分
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        # 抠代码 + 跑测试
        code = extract_code(response)
        if not code:
            passed = False
            no_code_count += 1
        else:
            passed = run_humaneval_test(
                code=code,
                test_code=sample["verification_info"]["test_code"],
                entry_point=sample["verification_info"]["entry_point"],
            )

        if passed:
            passed_count += 1

        results.append({
            "task_id": sample["task_id"],
            "passed": passed,
            "has_code": bool(code),
            "response": response,
            "code": code,
        })

    # === 总结 ===
    pass_rate = passed_count / len(ds)
    summary = {
        "model": model_name,
        "lora_path": lora_path,
        "label": label,
        "dataset": "openai/openai_humaneval",
        "split": "test",
        "evaluation_scope": "in_sample_same_tasks",
        "num_samples": len(ds),
        "passed": passed_count,
        "no_code": no_code_count,
        "pass_at_1": pass_rate,
        "results": results,
    }

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"eval_{label}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"评测完成! ({label})")
    print(f"{'=' * 60}")
    print(f"模型:        {model_name}")
    if lora_path:
        print(f"LoRA:        {lora_path}")
    print("范围:        in-sample (与项目训练使用相同 HumanEval 任务)")
    print(f"题数:        {len(ds)}")
    print(f"通过:        {passed_count}")
    print(f"无代码:      {no_code_count}")
    print(f"Pass@1:      {pass_rate:.2%}  ({passed_count}/{len(ds)})")
    print(f"详细结果:    {output_file}")
    print(f"{'=' * 60}\n")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="基座模型名/路径",
    )
    parser.add_argument(
        "--lora_path",
        default=None,
        help="LoRA adapter 路径 (留空 = 评基座)",
    )
    parser.add_argument(
        "--num_samples",
        type=_positive_int,
        default=164,
        help="评多少题，必须大于 0 (默认全部 164)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--label",
        default="baseline",
        help="结果标签 (用于命名输出文件)",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/eval",
    )
    args = parser.parse_args()

    evaluate(
        model_name=args.model,
        lora_path=args.lora_path,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        label=args.label,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
