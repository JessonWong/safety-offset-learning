model_name="meta-llama/Llama-3.1-8B-Instruct"

source env/bin/activate

export CUDA_VISIBLE_DEVICES=0
log_file="log/identity/filter.log"
mkdir -p log/identity
> $log_file

for seed in 7 42 89 16 63; do
    echo "===== Running with seed $seed =====" | tee -a "$log_file"

    out_dir="./finetune_model/identity/filter/seed_$seed"

    python -u -m safeoff.training.sft \
        --base_model_path "$model_name" \
        --output_dir "$out_dir" \
        --batch_size 4 \
        --gradient_accumulation_steps 2 \
        --epochs 5 \
        --save_steps 200 \
        --learning_rate 3e-4 \
        --poison_training_num 0 \
        --identity_shift_num 100 \
        --utility_training_num 1000 \
        --use_lora 1 \
        --filter 1 \
        --random_seed "$seed" \
        >> $log_file 2>&1

    python -u -m safeoff.evaluation.checkpoints \
        --base_model_path "$model_name" \
        --output_dir "$out_dir" \
        --checkpoint -1 \
        --batch_size 16 \
        --identity_shift 1 \
        --use_lora 1 \
        >> $log_file 2>&1

done
