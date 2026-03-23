#!/usr/bin/env python3
"""
Evaluate a LoRA SFT model on NIH Stage JSONL.

Outputs:
- eval_loss / perplexity
- simple predicted_stage exact-match rate (string normalized)
"""
from __future__ import annotations

import argparse
import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LoRA SFT model")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/train_data/nih_stage_model_sft_instruction_data.jsonl",
    )
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--max_eval_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)


def format_prompt_answer(rec: Dict[str, Any]) -> Tuple[str, str]:
    system = str(rec.get("system", "")).strip()
    instruction = str(rec.get("instruction", "")).strip()
    user_input = str(rec.get("input", "")).strip()
    output = rec.get("output", "")
    answer = safe_json_dumps(output) if not isinstance(output, str) else str(output)
    prompt = (
        "<|system|>\n"
        f"{system}\n\n"
        "<|user|>\n"
        f"{instruction}\n\n"
        f"{user_input}\n\n"
        "<|assistant|>\n"
    )
    return prompt, answer.strip()


def load_rows(path: str, limit: int) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
            if len(out) >= limit:
                break
    return out


def extract_predicted_stage(text: str) -> Optional[str]:
    # Try strict JSON first
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "predicted_stage" in obj:
            return str(obj["predicted_stage"]).strip().lower()
    except Exception:
        pass

    # Fallback: regex within generated text
    m = re.search(r'"predicted_stage"\s*:\s*"([^"]+)"', text)
    if m:
        return m.group(1).strip().lower()
    return None


def main() -> None:
    args = parse_args()
    device = choose_device()
    print(f"[INFO] device={device}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, args.lora_path)
    model.to(device)
    model.eval()

    rows = load_rows(args.data_path, args.max_eval_samples)
    texts = []
    prompts = []
    refs = []
    for r in rows:
        prompt, answer = format_prompt_answer(r)
        prompts.append(prompt)
        refs.append(answer)
        texts.append(prompt + answer)

    dataset = Dataset.from_dict({"text": texts})

    def tokenize_fn(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length, padding=False)

    eval_ds = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir="lora/outputs/eval_tmp",
            per_device_eval_batch_size=args.batch_size,
            report_to=[],
        ),
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )
    metrics = trainer.evaluate()
    eval_loss = metrics.get("eval_loss")
    ppl = math.exp(eval_loss) if eval_loss is not None and eval_loss < 20 else float("inf")

    # generation-based stage match
    match = 0
    valid = 0
    for prompt, ref in zip(prompts, refs):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        gen = model.generate(
            **inputs,
            max_new_tokens=220,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
        )
        out = tokenizer.decode(gen[0], skip_special_tokens=True)
        pred_stage = extract_predicted_stage(out)
        ref_stage = extract_predicted_stage(ref)
        if pred_stage is None or ref_stage is None:
            continue
        valid += 1
        if pred_stage == ref_stage:
            match += 1

    stage_match_acc = (match / valid) if valid > 0 else None

    result = {
        "eval_loss": eval_loss,
        "perplexity": ppl,
        "stage_match_acc": stage_match_acc,
        "stage_match_valid_samples": valid,
        "total_eval_samples": len(rows),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

