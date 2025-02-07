enhance_neuron_num=0
enhance_strength=1
export PYTHONPATH="$(pwd)"

gpu_id=$1
run_id=$2

adversarial_method_name_array=(
    "DAP"
    "DeepInception"
    "Jailbroken"
    "SafeEditWithAdv"
    "MultiJail"
    # "VirtualPersonas"
)
model_name="gemma-2b-it"
for adversarial_method_name in ${adversarial_method_name_array[@]}; do
    echo "$model_to_val_path $model_name $adversarial_method_name"
    trained_path="saved_models/itm/${run_id}/sft_${adversarial_method_name}_${model_name}_neuron_${run_id}"
    model_to_val_path=$(find ${trained_path} -mindepth 1 -maxdepth 1 -type d | head -n 1)
    /data/dans/anaconda3/envs/trl_new/bin/python src/4_edit_and_evaluate_harmless.py \
        --model_to_val_path $model_to_val_path \
        --model_name $model_name \
        --adversarial_method_name $adversarial_method_name \
        --generation_dir eval_results/harmless/$run_id/outputs_$run_id \
        --metric_dir eval_results/harmless/$run_id/metrics_$run_id \
        --gpu_id $gpu_id \
        --enhance_neuron_num ${enhance_neuron_num} \
        --enhance_strength ${enhance_strength} \
        --generate_batch_size 4 \
        --suffix_for_output neuron
        # --few_data_for_debug
done


adversarial_method_name_array=(
    "DAP"
    "DeepInception"
    "Jailbroken"
    "SafeEditWithAdv"
    "MultiJail"
    # "VirtualPersonas"
)
model_name="llama2-7b-chat"
for adversarial_method_name in ${adversarial_method_name_array[@]}; do
    echo "$model_to_val_path $model_name $adversarial_method_name"
    trained_path="saved_models/itm/${run_id}/sft_${adversarial_method_name}_${model_name}_neuron_${run_id}"
    model_to_val_path=$(find ${trained_path} -mindepth 1 -maxdepth 1 -type d | head -n 1)
    /data/dans/anaconda3/envs/trl_new/bin/python src/4_edit_and_evaluate_harmless.py \
        --model_to_val_path $model_to_val_path \
        --model_name $model_name \
        --adversarial_method_name $adversarial_method_name \
        --generation_dir eval_results/harmless/$run_id/outputs_$run_id \
        --metric_dir eval_results/harmless/$run_id/metrics_$run_id \
        --gpu_id $gpu_id \
        --enhance_neuron_num ${enhance_neuron_num} \
        --enhance_strength ${enhance_strength} \
        --generate_batch_size 4 \
        --suffix_for_output neuron
        # --few_data_for_debug
done