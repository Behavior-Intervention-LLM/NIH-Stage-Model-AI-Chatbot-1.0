#!/usr/bin/env python3
"""
LoRA SFT training script for NIH Stage Model JSONL data.

Expected JSONL schema (one record per line):
{
  "system": "...",
  "instruction": "...",
  "input": "...",
  "output": {... or "..."}
}
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LoRA SFT on NIH Stage JSONL")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/train_data/nih_stage_model_sft_instruction_data.jsonl",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HF model id or local base model path",
    )
    parser.add_argument("--output_dir", type=str, default="lora/outputs/qwen2.5-7b-lora")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--sample_eval_examples", type=int, default=3)

    # LoRA hyper-parameters
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    return parser.parse_args()


def _safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)

# convert data sturcture
def format_example(record: Dict[str, Any]) -> str:
    system = str(record.get("system", "")).strip()
    instruction = str(record.get("instruction", "")).strip()
    user_input = str(record.get("input", "")).strip()
    output = record.get("output", "")

    assistant_answer = _safe_json_dumps(output) if not isinstance(output, str) else output
    assistant_answer = assistant_answer.strip()

    # Plain instruction format that works across most base instruct models.
    prompt = (
        "<|system|>\n"
        f"{system}\n\n"
        "<|user|>\n"
        f"{instruction}\n\n"
        f"{user_input}\n\n"
        "<|assistant|>\n"
        f"{assistant_answer}"
    )
    return prompt

# load training data
def load_jsonl_dataset(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i}: {e}") from e
    if not rows:
        raise ValueError(f"No training rows found in {path}")
    return rows


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class PreparedData:
    train_ds: Dataset
    eval_ds: Dataset
    raw_eval_texts: List[str]

# train val test split
def build_datasets(rows: List[Dict[str, Any]], train_ratio: float, seed: int) -> PreparedData:
    all_texts = [format_example(x) for x in rows]
    random.Random(seed).shuffle(all_texts)
    split_idx = max(1, int(len(all_texts) * train_ratio))
    split_idx = min(split_idx, len(all_texts) - 1)
    train_texts = all_texts[:split_idx]
    eval_texts = all_texts[split_idx:]
    return PreparedData(
        train_ds=Dataset.from_dict({"text": train_texts}),
        eval_ds=Dataset.from_dict({"text": eval_texts}),
        raw_eval_texts=eval_texts,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # data preparation
    rows = load_jsonl_dataset(args.data_path)
    prepared = build_datasets(rows, args.train_ratio, args.seed)
    device = choose_device()
    print(f"[INFO] device={device}, train={len(prepared.train_ds)}, eval={len(prepared.eval_ds)}")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cuda":
        torch_dtype = torch.float16
    elif device == "mps":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # load base model
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.to(device)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, # generation task
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    # preparation for the LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def tokenize_fn(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding=False,
        )

    train_tokenized = prepared.train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    eval_tokenized = prepared.eval_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        report_to=[],
        bf16=False,
        fp16=(device == "cuda"),
        dataloader_num_workers=0,
        save_total_limit=2,
        load_best_model_at_end=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()
    eval_loss = eval_metrics.get("eval_loss")
    perplexity = math.exp(eval_loss) if eval_loss is not None and eval_loss < 20 else float("inf")
    eval_metrics["eval_perplexity"] = perplexity

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_metrics": train_result.metrics,
                "eval_metrics": eval_metrics,
                "config": vars(args),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"[INFO] metrics saved to {metrics_path}")

    # Quick qualitative eval samples
    sample_out_path = os.path.join(args.output_dir, "sample_generations.jsonl")
    model.eval()
    with torch.no_grad(), open(sample_out_path, "w", encoding="utf-8") as f:
        n = min(args.sample_eval_examples, len(prepared.raw_eval_texts))
        for i in range(n):
            text = prepared.raw_eval_texts[i]
            prompt = text.split("<|assistant|>\n")[0] + "<|assistant|>\n"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            gen = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.0,
                eos_token_id=tokenizer.eos_token_id,
            )
            decoded = tokenizer.decode(gen[0], skip_special_tokens=True)
            f.write(json.dumps({"idx": i, "prompt": prompt, "prediction": decoded}, ensure_ascii=False) + "\n")
    print(f"[INFO] sample generations saved to {sample_out_path}")
    print("[DONE] LoRA SFT training + eval finished.")


if __name__ == "__main__":
    main()

