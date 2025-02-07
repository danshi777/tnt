# accelerate launch src/2_sft_llama2.py
# ############ neuron training ############
gpu_id=$1

# gemma-2b-it
# each epoch
export PYTHONPATH="$(pwd)"
adversarial_method_name="Jailbroken"
model_name="gemma-2b-it"
output_dir="saved_models/${model_name}/itm/sft_neuron_${adversarial_method_name}_2"
mkdir -p $output_dir
CUDA_VISIBLE_DEVICES="${gpu_id}" /data/dans/anaconda3/envs/trl_new/bin/python src/2_sft_llama2.py \
    --model_id_or_path="/data/cordercorder/pretrained_models/google/gemma-2b-it" \
    --model_name=$model_name \
    --adversarial_method_name=$adversarial_method_name \
    --output_dir=$output_dir \
    --use_neuron_training \
    --trainable_param_num=230400 \
    --neuron_file="results/neurons/Jailbroken_gemma-2b-it-cross_entropy-intermediate-1_top50_neuron.json" \
    --get_intermediate_size_or_hidden_size="intermediate" \
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



adversarial_method_name="SafeEditWithAdv"
model_name="gemma-2b-it"
output_dir="saved_models/${model_name}/itm/sft_neuron_${adversarial_method_name}_2"
mkdir -p $output_dir
CUDA_VISIBLE_DEVICES="${gpu_id}" /data/dans/anaconda3/envs/trl_new/bin/python src/2_sft_llama2.py \
    --model_id_or_path="/data/cordercorder/pretrained_models/google/gemma-2b-it" \
    --model_name=$model_name \
    --adversarial_method_name=$adversarial_method_name \
    --output_dir=$output_dir \
    --use_neuron_training \
    --trainable_param_num=230400 \
    --neuron_file="results/neurons/SafeEditWithAdv_gemma-2b-it-cross_entropy-intermediate-1_top50_neuron.json" \
    --get_intermediate_size_or_hidden_size="intermediate" \
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