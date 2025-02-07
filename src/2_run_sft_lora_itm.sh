# accelerate launch src/2_sft_llama2.py
# ############ lora training ############
export PYTHONPATH="$(pwd)"

gpu_id=$1
model_name=$2
adversarial_method_name=$3
rank=$4
run_id=$5
let alpha=rank*2

echo "gpu_id=$gpu_id, rank=$rank, alpha=$alpha"
# each epoch
output_dir="saved_models/itm/${model_name}/sft_${adversarial_method_name}_lora_${run_id}"
mkdir -p $output_dir
CUDA_VISIBLE_DEVICES="${gpu_id}" CUDA_LAUNCH_BLOCKING="1" /data/dans/anaconda3/envs/trl_new/bin/python src/2_sft_llama2.py \
    --model_name=$model_name \
    --adversarial_method_name=$adversarial_method_name \
    --num_of_examples=100 \
    --output_dir=$output_dir \
    --use_peft \
    --lora_r=$rank \
    --lora_alpha=$alpha \
    --lora_dropout=0.0 \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --gradient_accumulation_steps=2 \
    --gradient_checkpointing=False \
    --eval_strategy="epoch" \
    --logging_strategy="epoch" \
    --save_strategy="no" \
    --save_total_limit=0 \
    --load_best_model_at_end=False \
    --num_train_epochs=10 \
    --learning_rate=1e-4 \
    --lr_scheduler_type="cosine" \
    --warmup_steps=100 \
    --weight_decay=0.05 \
    --optim="paged_adamw_32bit" \
    --report_to="wandb" \
    2>&1 | tee -a $output_dir/log.log
