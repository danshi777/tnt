enhance_neuron_num=0
enhance_strength=1
export PYTHONPATH="$(pwd)"

# 运行：
# src/4_run_test_harmless_selfreminder_chat_models_itm.sh 1 harmless-selfreminder

gpu_id=$1
run_id=$2
adversarial_method_name_array=(
    "SelfReminder"
)

# model_to_val_path=/data2/dans/models/vicuna-7b-v1.5
# model_name=vicuna-7b-v1.5

model_name=gemma-2b-it
model_to_val_path=/data/cordercorder/pretrained_models/google/gemma-2b-it

for adversarial_method_name in ${adversarial_method_name_array[@]}; do
    echo "Self Reminder $model_to_val_path $model_name $adversarial_method_name"
    /data/dans/anaconda3/envs/trl_new/bin/python src/4_edit_and_evaluate_harmless.py \
        --model_to_val_path $model_to_val_path \
        --model_name $model_name \
        --adversarial_method_name $adversarial_method_name \
        --generation_dir eval_results/$run_id/outputs_$run_id \
        --metric_dir eval_results/$run_id/metrics_$run_id \
        --gpu_id $gpu_id \
        --enhance_neuron_num ${enhance_neuron_num} \
        --enhance_strength ${enhance_strength} \
        --generate_batch_size 4 \
        --suffix_for_output vanilla
        # --few_data_for_debug
done

model_name=llama2-7b-chat
model_to_val_path=/data1/dans/models/llama2-7b-chat-hf
for adversarial_method_name in ${adversarial_method_name_array[@]}; do
    echo "$model_to_val_path $model_name $adversarial_method_name"
    /data/dans/anaconda3/envs/trl_new/bin/python src/4_edit_and_evaluate_harmless.py \
        --model_to_val_path $model_to_val_path \
        --model_name $model_name \
        --adversarial_method_name $adversarial_method_name \
        --generation_dir eval_results/$run_id/outputs_$run_id \
        --metric_dir eval_results/$run_id/metrics_$run_id \
        --gpu_id $gpu_id \
        --enhance_neuron_num ${enhance_neuron_num} \
        --enhance_strength ${enhance_strength} \
        --generate_batch_size 2 \
        --suffix_for_output vanilla
        # --few_data_for_debug
done