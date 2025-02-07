# accelerate launch src/2_sft_llama2.py
# ############ lora training ############

gpu_id=$1
# gemma-2b-it
# each epoch
export PYTHONPATH="$(pwd)"
adversarial_method_name="Jailbroken"
model_name="gemma-2b-it"
output_dir="saved_models/${model_name}/sft_lora_${adversarial_method_name}_2"
mkdir -p $output_dir
CUDA_VISIBLE_DEVICES="${gpu_id}" /data/dans/anaconda3/envs/trl_new/bin/python src/2_sft_llama2.py \
    --model_id_or_path="/data/cordercorder/pretrained_models/google/gemma-2b-it" \
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
    --eval_strategy="epoch" \
    --logging_strategy="epoch" \
    --save_strategy="epoch" \
    --save_total_limit=1 \
    --load_best_model_at_end=True \
    --num_train_epochs=8 \
    --learning_rate=1e-4 \
    --lr_scheduler_type="cosine" \
    --warmup_steps=100 \
    --weight_decay=0.05 \
    --optim="paged_adamw_32bit" \
    --report_to="wandb" \
    2>&1 | tee -a $output_dir/log.log
