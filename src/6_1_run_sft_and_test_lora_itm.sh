# accelerate launch src/2_sft_llama2.py
# ############ lora training ############
export PYTHONPATH="$(pwd)"

# 运行：
# 参数各是：gpu_id model_name adversarial_method_name rank run_id
# src/6_1_run_sft_and_test_lora_itm.sh 2 "gemma-2b-it" "VirtualPersonas" 2 test0123

gpu_id=$1
model_name=$2
adversarial_method_name=$3
rank=$4
run_id=$5
let alpha=rank*2

echo "--- run_sft_and_test_lora, ${model_name}, ${adversarial_method_name}, gpu_id=$gpu_id, rank=$rank, alpha=$alpha ---"
output_dir="saved_models/itm/${run_id}/sft_${adversarial_method_name}_${model_name}_lora_${run_id}"
mkdir -p $output_dir
CUDA_VISIBLE_DEVICES="${gpu_id}" CUDA_LAUNCH_BLOCKING="1" /data/dans/anaconda3/envs/trl_new/bin/python src/2_sft_and_test.py \
    --model_name=$model_name \
    --adversarial_method_name=$adversarial_method_name \
    --num_of_examples=2 \
    --output_dir=$output_dir \
    --use_peft \
    --lora_r=$rank \
    --lora_alpha=$alpha \
    --lora_dropout=0.01 \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --gradient_accumulation_steps=2 \
    --gradient_checkpointing=False \
    --eval_strategy="epoch" \
    --logging_strategy="steps" \
    --logging_steps=1 \
    --save_strategy="epoch" \
    --save_total_limit=1 \
    --load_best_model_at_end=True \
    --num_train_epochs=1 \
    --learning_rate=2e-5 \
    --lr_scheduler_type="cosine" \
    --warmup_steps=15 \
    --weight_decay=0.05 \
    --optim="rmsprop" \
    --report_to="wandb" \
    --generation_dir="eval_results/$run_id/outputs_$run_id" \
    --metric_dir="eval_results/$run_id/metrics_$run_id" \
    --generate_batch_size=4 \
    --suffix_for_output="lora" \
    --few_data_for_debug \
    2>&1 | tee -a $output_dir/log.log

    # --save_strategy="epoch" \
    # --save_total_limit=1 \
    # --load_best_model_at_end=True \