# LoRA SFT (Qwen2.5-7B)

## 1) Install training dependencies

```bash
pip install -r lora/requirements.txt
```

## 2) Start LoRA SFT training

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

After training, the script produces:
- `metrics.json` (includes train/eval metrics and perplexity)
- `sample_generations.jsonl` (qualitative evaluation generations)

## 3) Evaluate a trained LoRA adapter only

```bash
python lora/evaluate_lora_sft.py \
  --data_path data/train_data/nih_stage_model_sft_instruction_data.jsonl \
  --base_model Qwen/Qwen2.5-7B-Instruct \
  --lora_path lora/outputs/qwen2.5-7b-lora \
  --max_eval_samples 100
```

Outputs include:
- `eval_loss`
- `perplexity`
- `stage_match_acc` (simple exact-match rate based on `predicted_stage`)

## 4) Notes for Mac

- 7B can be memory-heavy on Mac. Consider:
  - `--max_length 1024`
  - `--gradient_accumulation_steps 16`
  - or validate the training pipeline with a smaller base model first
