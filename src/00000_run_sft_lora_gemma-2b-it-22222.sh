# accelerate launch src/2_sft_llama2.py
# ############ lora training ############

# 运行：
# src/00000_run_sft_lora_gemma-2b-it-22222.sh gpu_id run_id
# src/00000_run_sft_lora_gemma-2b-it-22222.sh 0 test222

gpu_id=$1
run_id=$2
# model_name=$3
# adversarial_method_name=$4

# each epoch
export PYTHONPATH="$(pwd)"
adversarial_method_name="Jailbroken"
model_name="gemma-2b-it"
output_dir="saved_models/${model_name}/itm/sft_lora_${adversarial_method_name}_${run_id}"
mkdir -p $output_dir
CUDA_VISIBLE_DEVICES="${gpu_id}" CUDA_LAUNCH_BLOCKING="1" /data/dans/anaconda3/envs/trl_new/bin/python src/2_sft_llama2.py \
    --model_name=$model_name \
    --adversarial_method_name=$adversarial_method_name \
    --output_dir=$output_dir \
    --use_peft \
    --lora_r=2 \
    --lora_alpha=2 \
    --lora_dropout=0.0 \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --gradient_accumulation_steps=2 \
    --gradient_checkpointing=False \
    --eval_strategy="no" \
    --logging_strategy="epoch" \
    --save_strategy="no" \
    --save_total_limit=1 \
    --load_best_model_at_end=True \
    --num_train_epochs=500 \
    --learning_rate=1e-4 \
    --lr_scheduler_type="cosine" \
    --warmup_steps=100 \
    --weight_decay=0.05 \
    --optim="paged_adamw_32bit" \
    --report_to="wandb" \
    | tee -a $output_dir/log.log
