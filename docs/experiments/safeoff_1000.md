# SafeOff regeneration check

This protocol regenerates SafeOff data for the first 1,000 Alpaca rows,
fine-tunes `NousResearch/Meta-Llama-3.1-8B-Instruct` on it for 300 optimizer
steps, and measures keyword refusal on AdvBench before and after. It was run on
2026-10-03 at commit `a2cf8ccc0bb2a6784b3cf9e66ae7e8fe02e03bcb` to check that the
released code reproduces the main effect without the historical data files.

## 1. Generate data

Run the [SafeOff generation](../data.md#safeoff-generation) command with
`--output-dir outputs/safeoff_llama31_8b --run-dir outputs/safeoff_llama31_8b`,
and build `data/advbench.csv` as described in
[AdvBench prompts](../data.md#advbench-prompts). Generation uses two GPUs; the
steps below use one.

## 2. Fine-tune

```bash
RUN=outputs/safeoff_llama31_8b
safeoff train \
  --base_model_path NousResearch/Meta-Llama-3.1-8B-Instruct \
  --output_dir "$RUN/training" \
  --utility_dataset_config utility_dataset_config_alpaca \
  --utility_take_first_n 1000 --utility_training_num 1000 \
  --poison_training_num 0 --identity_shift_num 0 --backdoor 0 \
  --augment_utility_with_lat 0 \
  --advprompt_suffix "$RUN/reverse_eval_prompt_replacement_train_compat.csv" \
  --advprompt_use_suffix 0 --advprompt_training_num 1000 \
  --advprompt_clean_training_num 0 \
  --batch_size 1 --gradient_accumulation_steps 4 --epochs 1 \
  --max_steps 300 --save_steps 25 --save_checkpoint_zero 1 \
  --learning_rate 0.0001 --weight_decay 0.01 \
  --use_lora 1 --lora_r 8 --lora_alpha 32 --lora_dropout 0.05 \
  --lora_target_modules q_proj v_proj \
  --random_seed 24 --data_random_seed 24 --lora_init_seed 24 \
  --max_sequence_length 1024 --fail_on_overlength_examples 1 \
  --evaluation_strategy no --filter 0 --save_grads_every 0 \
  --use_cache 0 --use_device_map 0 --bf16 1
```

The effective batch size is 4, so 300 steps cover 1.2 epochs. Training uses the
system prompt `You are a helpful assistant.`, keeps every original Alpaca
answer, and adds no clean rows. The Trainer seed stays at its default of 42. The
recorded run also passed `--utility_dataset_path` with a local JSON export of
the same 1,000 rows for an offline node; without it, the preset loads them from
the Hub.

## 3. Evaluate

```bash
RUN=outputs/safeoff_llama31_8b
safeoff eval system-ablation --csv-path data/advbench.csv \
  --base-model NousResearch/Meta-Llama-3.1-8B-Instruct \
  --output-dir "$RUN/evaluation/base" \
  --conditions helpful_system --batch-size 8 --max-new-tokens 200 --dtype bfloat16
safeoff eval system-ablation --csv-path data/advbench.csv \
  --base-model NousResearch/Meta-Llama-3.1-8B-Instruct \
  --adapter "$RUN/training/Meta-Llama-3.1-8B-Instruct/model/checkpoint-300" \
  --output-dir "$RUN/evaluation/finetuned" \
  --conditions helpful_system --batch-size 8 --max-new-tokens 200 --dtype bfloat16
```

Each output directory has `summary.json`, with the rate under
`conditions.helpful_system.refusal_rate`, and `helpful_system.json`, with every
response.

## Recorded result

| Model | AdvBench keyword refusal |
| --- | --- |
| Base | 99.23% (516/520) |
| After 300 SafeOff steps | 6.35% (33/520) |

Keyword refusal is a heuristic, not a harmfulness score. This run had no hosted
judge configured and applied no WildGuard filtering. The regenerated training
prompts differ from the historical `data/8b_newtarget.csv`; see
[data inputs](../data.md#safeoff-generation).
