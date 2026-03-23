# LoRA SFT (Qwen2.5-7B)

## 1) 安装训练依赖

```bash
pip install -r lora/requirements.txt
```

## 2) 开始 LoRA SFT 训练

```bash
python lora/train_lora_sft.py \
  --data_path data/train_data/nih_stage_model_sft_instruction_data.jsonl \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --output_dir lora/outputs/qwen2.5-7b-lora \
  --num_train_epochs 2 \
  --learning_rate 2e-4 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8
```

训练完成后会生成：
- `metrics.json`（包含 train/eval 指标，含 perplexity）
- `sample_generations.jsonl`（评估样例生成）

## 3) 单独评估已训练 LoRA

```bash
python lora/evaluate_lora_sft.py \
  --data_path data/train_data/nih_stage_model_sft_instruction_data.jsonl \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --lora_path lora/outputs/qwen2.5-7b-lora \
  --max_eval_samples 100
```

会输出：
- `eval_loss`
- `perplexity`
- `stage_match_acc`（基于 `predicted_stage` 的简单精确匹配率）

## 4) Mac 注意事项

- 7B 在 Mac 上可能内存压力大，建议先减小：
  - `--max_length 1024`
  - `--gradient_accumulation_steps 16`
  - 或先用更小基座模型验证流程
