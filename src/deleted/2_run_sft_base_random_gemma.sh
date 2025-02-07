# accelerate launch src/2_sft_llama2.py
# ############ random neurons training ############
# gemma-2b
# each epoch
export PYTHONPATH="$(pwd)"
output_dir="saved_models/gemma/sft_random_safeedit_gemma-2b_5.3"
mkdir -p $output_dir
CUDA_VISIBLE_DEVICES="2" /data/dans/anaconda3/envs/trl_new/bin/python src/2_sft_llama2.py \
    --model_id_or_path="/data1/dans/models/gemma-2b" \
    --data_dir="/data/dans/projects/trl_new/data/SafeEdit/train/" \
    --model_name="gemma-2b" \
    --data_name="safeedit" \
    --output_dir=$output_dir \
    --use_random_training \
    --seed=3 \
    --trainable_param_num=230400 \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --gradient_accumulation_steps=2 \
    --gradient_checkpointing=False \
    --eval_strategy="epoch" \
    --logging_strategy="epoch" \
    --logging_dir="$output_dir/logging.log" \
    --save_strategy="epoch" \
    --save_total_limit=1 \
    --load_best_model_at_end=True \
    --num_train_epochs=8 \
    --learning_rate=5e-5 \
    --lr_scheduler_type="cosine" \
    --warmup_steps=100 \
    --weight_decay=0.05 \
    --optim="paged_adamw_32bit" \
    --report_to="wandb" \
    2>&1 | tee -a $output_dir/log.log
