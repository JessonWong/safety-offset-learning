model_name="meta-llama/Llama-3.1-8B-Instruct"

source env/bin/activate

export CUDA_VISIBLE_DEVICES=1
log_file="log/LATharm/filter.log"
mkdir -p log/LATharm
> $log_file

python -u -m safeoff.training.sft \
    --base_model_path "$model_name" \
    --output_dir ./finetune_model/LATharm/filter \
    --batch_size 4 \
    --gradient_accumulation_steps 2 \
    --epochs 3 \
    --save_steps 200 \
    --learning_rate 1e-4 \
    --poison_training_num 100 \
    --identity_shift_num 0 \
    --utility_training_num 1000 \
    --use_lora 1 \
    --filter 1 \
    >> $log_file 2>&1

python -u -m safeoff.evaluation.checkpoints \
    --base_model_path "$model_name" \
    --output_dir ./finetune_model/LATharm/filter \
    --checkpoint -1 \
    --batch_size 16 \
    --use_lora 1 \
    >> $log_file 2>&1
