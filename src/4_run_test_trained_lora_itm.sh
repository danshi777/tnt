enhance_neuron_num=0
enhance_strength=1

gpu_id=$1
model_name=$2
adversarial_method_name=$3
    # Jailbroken
    # SafeEditWithAdv
run_id=$4

model_to_val_path="saved_models/itm/${model_name}/sft_${adversarial_method_name}_lora_${run_id}"

# random
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
    --suffix_for_output lora
    # --few_data_for_debug
