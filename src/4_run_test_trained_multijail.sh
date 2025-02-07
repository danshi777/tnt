export PYTHONPATH="$(pwd)"
enhance_neuron_num=0
enhance_strength=1

gpu_id=$1
model_name="gemma-2b-it"
adversarial_method_name="MultiJail"
    # Jailbroken
    # SafeEditWithAdv

for run_id in 13 14 15; do
    trained_path="saved_models/itm/${run_id}/sft_${adversarial_method_name}_${model_name}_lora_${run_id}"
    model_to_val_path=$(find ${trained_path} -mindepth 1 -maxdepth 1 -type d | head -n 1)
    echo "$model_to_val_path $model_name $adversarial_method_name"
    /data/dans/anaconda3/envs/trl_new/bin/python src/4_edit_and_evaluate_toxic.py \
        --model_to_val_path $model_to_val_path \
        --model_name $model_name \
        --adversarial_method_name $adversarial_method_name \
        --generation_dir eval_results2/$run_id/outputs_$run_id \
        --metric_dir eval_results2/$run_id/metrics_$run_id \
        --gpu_id $gpu_id \
        --enhance_neuron_num ${enhance_neuron_num} \
        --enhance_strength ${enhance_strength} \
        --generate_batch_size 4 \
        --suffix_for_output lora
        # --few_data_for_debug
done

for run_id in 13 14 15; do
    trained_path="saved_models/itm/${run_id}/sft_${adversarial_method_name}_${model_name}_neuron_${run_id}"
    model_to_val_path=$(find ${trained_path} -mindepth 1 -maxdepth 1 -type d | head -n 1)
    echo "$model_to_val_path $model_name $adversarial_method_name"
    /data/dans/anaconda3/envs/trl_new/bin/python src/4_edit_and_evaluate_toxic.py \
        --model_to_val_path $model_to_val_path \
        --model_name $model_name \
        --adversarial_method_name $adversarial_method_name \
        --generation_dir eval_results2/$run_id/outputs_$run_id \
        --metric_dir eval_results2/$run_id/metrics_$run_id \
        --gpu_id $gpu_id \
        --enhance_neuron_num ${enhance_neuron_num} \
        --enhance_strength ${enhance_strength} \
        --generate_batch_size 4 \
        --suffix_for_output neuron
        # --few_data_for_debug
done