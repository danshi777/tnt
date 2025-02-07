enhance_neuron_num=0
enhance_strength=1
export PYTHONPATH="$(pwd)"

# 运行：
# src/4_run_cross_test_chat_models_itm.sh 4 cross_test

gpu_id=$1
run_id=$2
adversarial_method_name_array=(
    # "VirtualPersonas"
    # "DeepInception"
    # "DAP"
    # "Cipher"
    "MultiJail"
    # "Jailbroken"
    # "SafeEditWithAdv"
)

model_name="model-is-DAP-gemma-2b-it-neuron-14"
trained_path="saved_models/itm/14/sft_DAP_gemma-2b-it_neuron_14"
model_to_val_path=$(find ${trained_path} -mindepth 1 -maxdepth 1 -type d | head -n 1)
for adversarial_method_name in ${adversarial_method_name_array[@]}; do
    echo "$model_to_val_path $model_name $adversarial_method_name"
    /data/dans/anaconda3/envs/trl_new/bin/python src/4_edit_and_evaluate_toxic.py \
        --model_to_val_path $model_to_val_path \
        --model_name $model_name \
        --adversarial_method_name $adversarial_method_name \
        --generation_dir eval_results/$run_id/outputs_$run_id \
        --metric_dir eval_results/$run_id/metrics_$run_id \
        --gpu_id $gpu_id \
        --enhance_neuron_num ${enhance_neuron_num} \
        --enhance_strength ${enhance_strength} \
        --generate_batch_size 4 \
        --suffix_for_output ""
        # --few_data_for_debug
done