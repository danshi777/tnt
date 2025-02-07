# accelerate launch src/2_sft_llama2.py
# ############ neuron training ############
export PYTHONPATH="$(pwd)"

gpu_id=$1
model_name=$2
adversarial_method_name=$3
trainable_param_num=$4
    # r=2,gemma-2b: 230400
    # r=2,llama2-7b: 1048576
    # r=4,gemma-2b: 460800
    # r=4,llama2-7b: 2097152
    # r=8,gemma-2b: 921600
    # r=8,llama2-7b: 4194304
run_id=$5
get_intermediate_size_or_hidden_size="intermediate"

echo "run this on 25.2.7"
echo "--- run_sft_and_test_neuron, ${model_name}, ${adversarial_method_name}, gpu_id=$gpu_id, top_n=$top_n ---"
output_dir="saved_models/itm/${run_id}/sft_${adversarial_method_name}_${model_name}_neuron_${run_id}"
mkdir -p $output_dir
CUDA_VISIBLE_DEVICES="${gpu_id}" /data/dans/anaconda3/envs/trl_new/bin/python src/2_sft_and_test.py \
    --model_name=$model_name \
    --adversarial_method_name=$adversarial_method_name \
    --num_of_examples 100 \
    --output_dir=$output_dir \
    --use_neuron_training \
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
    --generation_dir="eval_results/mix_neuron/$run_id/outputs_$run_id" \
    --metric_dir="eval_results/mix_neuron/$run_id/metrics_$run_id" \
    --generate_batch_size=4 \
    --suffix_for_output="neuron" \
    --mix_neuron=True \
    2>&1 | tee -a $output_dir/log.log

# 2>&1 | tee -a $output_dir/log.log
# | tee -a $output_dir/log.log

    # --save_strategy="epoch" \
    # --save_total_limit=1 \
    # --load_best_model_at_end=True \

# # llama3-8b
# # each epoch
# export PYTHONPATH="$(pwd)"
# output_dir="saved_models/llama3-8b/sft_neuron_safeedit_5"
# mkdir -p $output_dir
# CUDA_VISIBLE_DEVICES="4" /data/dans/anaconda3/envs/trl_new/bin/python src/2_sft_llama2.py \
#     --model_id_or_path="/home/sth/data/code/Meta-Llama-3-8B" \
#     --data_dir="/data/dans/projects/trl_new/data/SafeEdit/train/" \
#     --model_name="llama3-8b" \
#     --data_name="safeedit" \
#     --output_dir=$output_dir \
#     --use_neuron_training \
#     --trainable_param_num=1703936 \
#     --neuron_file="results/neurons/SafeEdit_llama3-8b-cross_entropy-intermediate-1_top30_neuron.json" \
#     --per_device_train_batch_size=4 \
#     --per_device_eval_batch_size=4 \
#     --gradient_accumulation_steps=2 \
#     --gradient_checkpointing=False \
#     --eval_strategy="epoch" \
#     --logging_strategy="epoch" \
#     --save_strategy="epoch" \
#     --save_total_limit=1 \
#     --load_best_model_at_end=True \
#     --num_train_epochs=6 \
#     --learning_rate=2e-5 \
#     --lr_scheduler_type="cosine" \
#     --warmup_steps=100 \
#     --weight_decay=0.05 \
#     --optim="paged_adamw_32bit" \
#     --report_to="wandb" \
#     2>&1 | tee -a $output_dir/log.log






# # each xxx steps
# export PYTHONPATH="$(pwd)"
# output_dir="saved_models/sft_neuron_safeedit_gemma-2b_4"
# mkdir -p $output_dir
# CUDA_VISIBLE_DEVICES="2" /data/dans/anaconda3/envs/trl_new/bin/python src/2_sft_llama2.py \
#     --model_id_or_path="/data1/dans/models/gemma-2b" \
#     --data_dir="/data/dans/projects/trl_new/data/SafeEdit/train/" \
#     --model_name="gemma-2b" \
#     --data_name="safeedit" \
#     --output_dir=$output_dir \
#     --use_neuron_training \
#     --trainable_param_num=230400 \
#     --neuron_file="results/neurons/SafeEdit_gemma-2b-cross_entropy-intermediate-1_neuron.json" \
#     --per_device_train_batch_size=4 \
#     --per_device_eval_batch_size=4 \
#     --gradient_accumulation_steps=2 \
#     --gradient_checkpointing=False \
#     --eval_strategy="steps" \
#     --eval_steps=100 \
#     --logging_steps=100 \
#     --save_strategy="steps" \
#     --save_total_limit=2 \
#     --load_best_model_at_end=True \
#     --metric_for_best_model="eval_loss" \
#     --greater_is_better=False, \
#     --num_train_epochs=8 \
#     --learning_rate=5e-5 \
#     --lr_scheduler_type="cosine" \
#     --warmup_steps=100 \
#     --weight_decay=0.05 \
#     --optim="paged_adamw_32bit" \
#     --report_to="wandb" \
#     2>&1 | tee -a $output_dir/log.log




# # llama3-8b
# export PYTHONPATH="$(pwd)"
# python src/2_sft_llama2.py \
#     --model_id_or_path="/home/sth/data/code/Meta-Llama-3-8B" \
#     --data_dir="/data/dans/projects/trl_new/data/SafeEdit/train/" \
#     --model_name="llama3-8b" \
#     --data_name="safeedit" \
#     --run_name_prefix="sft_neuron" \
#     --use_neuron_training \
#     --trainable_param_num=4194304 \
#     --neuron_file="results/neurons/SafeEdit_gemma-2b-cross_entropy-intermediate-1_neuron.json" \
#     --per_device_train_batch_size=4






# # Full training
# python examples/scripts/sft.py \
#     --model_name_or_path Qwen/Qwen2-0.5B \
#     --dataset_name trl-lib/Capybara \
#     --learning_rate 2.0e-5 \
#     --num_train_epochs 1 \
#     --packing \
#     --per_device_train_batch_size 2 \
#     --gradient_accumulation_steps 8 \
#     --gradient_checkpointing \
#     --logging_steps 25 \
#     --eval_strategy steps \
#     --eval_steps 100 \
#     --output_dir Qwen2-0.5B-SFT \
#     --push_to_hub

# # LoRA
# python examples/scripts/sft.py \
#     --model_name_or_path Qwen/Qwen2-0.5B \
#     --dataset_name trl-lib/Capybara \
#     --learning_rate 2.0e-4 \
#     --num_train_epochs 1 \
#     --packing \
#     --per_device_train_batch_size 2 \
#     --gradient_accumulation_steps 8 \
#     --gradient_checkpointing \
#     --logging_steps 25 \
#     --eval_strategy steps \
#     --eval_steps 100 \
#     --use_peft \
#     --lora_r 32 \
#     --lora_alpha 16 \
#     --output_dir Qwen2-0.5B-SFT \
#     --push_to_hub