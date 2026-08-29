"""
GRPO 训练脚本 —— Qwen2.5-Coder-1.5B 在 HumanEval 上用 GRPO 训练

设计约束:
  - 4070 Ti SUPER 16GB
  - LoRA 微调 (不是全参数)
  - bf16 精度 + gradient checkpointing 省显存

运行方式:
    # 先跑 debug 模式验证不崩 (只跑 2 步, ~5 分钟)
    python src/train.py --debug

    # 正式训练 (全数据, 2 epochs, v3 最优配方)
    python src/train.py

    # 自定义步数
    python src/train.py --max_steps 100
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig

# 让我们能 import 同目录的模块
sys.path.insert(0, str(Path(__file__).parent))
from data_prep import load_humaneval
from reward_funcs import code_reward_humaneval_partial, format_reward


def _parse_max_steps(value: str) -> int:
    """Accept the Trainer sentinel (-1) or a positive step limit."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be -1 or a positive integer"
        ) from exc
    if parsed == -1 or parsed > 0:
        return parsed
    raise argparse.ArgumentTypeError("must be -1 or a positive integer")


def main(args):
    # =========================================================================
    # Step 1: 加载模型 + tokenizer
    # =========================================================================
    print(f"[1/5] 加载模型: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,      # bf16 省一半显存
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {param_count / 1e9:.2f}B")

    # =========================================================================
    # Step 2: 加载数据
    # =========================================================================
    print(f"\n[2/5] 加载 HumanEval")
    ds = load_humaneval("test")
    if args.debug:
        ds = ds.select(range(4))         # debug 只用 4 道题
    print(f"  训练集大小: {len(ds)}")

    # =========================================================================
    # Step 3: LoRA 配置 (只训 adapter, 基座冻结)
    # =========================================================================
    print(f"\n[3/5] 配置 LoRA")
    peft_config = LoraConfig(
        r=16,                             # LoRA rank, 越大能力越强但显存越多
        lora_alpha=32,                    # 缩放因子, 通常是 r 的 2 倍
        target_modules=[                  # 在哪些层加 LoRA
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
    )
    print(f"  LoRA rank={peft_config.r}, alpha={peft_config.lora_alpha}")

    # =========================================================================
    # Step 4: GRPO 训练配置
    # =========================================================================
    print(f"\n[4/5] 配置 GRPO 训练参数")
    training_args = GRPOConfig(
        output_dir=args.output_dir,

        # --- 学习率 (小心调, 大了会崩) ---
        learning_rate=1e-5,        # v3: 5e-6 → 1e-5, v2 信号好, 可加速
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,

        # --- GRPO 目标 ---
        beta=0.0,                         # 复现 v3: 不加载参考模型、不加 KL 惩罚

        # --- 小 GPU 关键设置 ---
        per_device_train_batch_size=1,    # 每卡 batch_size
        gradient_accumulation_steps=4,    # 累积 4 步 = 等效 batch 4
        num_generations=4,                # GRPO 每题生成 4 个候选 (组大小)
        max_prompt_length=512,
        max_completion_length=512,

        # --- 训练时长 ---
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,

        # --- 显存优化 ---
        bf16=True,
        gradient_checkpointing=True,      # 用时间换显存, 对 16GB 关键

        # --- 日志 + 保存 ---
        logging_steps=1,                  # 每步都打日志
        save_steps=50,
        save_total_limit=2,               # 只留最近 2 个 checkpoint
        report_to="none",                 # 不用 wandb (可改成 "wandb" 如果你装了)
    )

    # =========================================================================
    # Step 5: 构建 trainer 开始训练
    # =========================================================================
    print(f"\n[5/5] 构建 GRPOTrainer 并开始训练")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            code_reward_humaneval_partial,  # reward 1: 代码部分正确性 (0~1 连续)
            format_reward,                   # reward 2: 输出格式 (0 or 0.5)
        ],
        args=training_args,
        train_dataset=ds,
        peft_config=peft_config,
    )

    print("\n" + "=" * 60)
    print("开始训练! 观察以下指标:")
    print("  - reward:      平均奖励 (应该慢慢上升)")
    print("  - reward_std:  同组 completion 的奖励方差")
    print("  - loss:        损失值 (会从 0 开始涨, 这是正常的!)")
    print("=" * 60 + "\n")

    trainer.train()

    trainer.save_model(args.output_dir)
    print(f"\n[OK] 训练完成! 模型保存到: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="HuggingFace 模型名或本地路径",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/grpo_humaneval",
        help="模型保存目录",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="调试模式: 只用 4 道题, 验证不崩",
    )
    parser.add_argument(
        "--max_steps",
        type=_parse_max_steps,
        default=-1,
        help="最多跑多少步，必须为正整数或 -1 (-1 = 全跑)",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=2,
        help="训练轮数 (默认 2，即实验中的 v3 最优配方)",
    )
    args = parser.parse_args()

    # debug 模式覆盖 max_steps
    if args.debug and args.max_steps == -1:
        args.max_steps = 2

    main(args)
