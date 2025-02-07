# accelerate launch src/2_sft_llama2.py
# ############ random neurons in last layer training ############
export PYTHONPATH="$(pwd)"

gpu_id=$1
model_name=$2
adversarial_method_name=$3
trainable_param_num=$4
random_seed=$5
run_id=$6

get_intermediate_size_or_hidden_size="intermediate"

echo "--- run_sft_and_test_random_last_layer, ${model_name}, ${adversarial_method_name}, gpu_id=$gpu_id, random_seed=$random_seed ---"
output_dir="saved_models/itm/${run_id}/randomll/sft_${adversarial_method_name}_${model_name}_randomll_seed${random_seed}_${run_id}"
mkdir -p $output_dir
CUDA_VISIBLE_DEVICES="${gpu_id}" /data/dans/anaconda3/envs/trl_new/bin/python src/2_sft_and_test.py \
    --model_name=$model_name \
    --adversarial_method_name=$adversarial_method_name \
    --num_of_examples=100 \
    --output_dir=$output_dir \
    --use_random_last_layer_training \
    --seed=$random_seed \
    --trainable_param_num=$trainable_param_num \
    --get_intermediate_size_or_hidden_size=$get_intermediate_size_or_hidden_size \
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
    --num_train_epochs=12 \
    --learning_rate=2e-5 \
    --lr_scheduler_type="cosine" \
    --warmup_steps=15 \
    --weight_decay=0.05 \
    --optim="rmsprop" \
    --report_to="wandb" \
    --generation_dir="eval_results/$run_id/outputs_$run_id" \
    --metric_dir="eval_results/$run_id/metrics_$run_id" \
    --generate_batch_size=4 \
    --suffix_for_output="randomll" \
    2>&1 | tee -a $output_dir/log.log