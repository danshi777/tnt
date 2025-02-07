enhance_neuron_num=0
enhance_strength=1

gpus=$1
model_name=$2
adversarial_method_name=$3
    # SafeEditWithAdv
    # Jailbroken
random_seed=$4
run_id=$5
model_to_val_path="saved_models/itm/${model_name}/randomll/sft_${adversarial_method_name}_seed${random_seed}_${run_id}"

# random last layer
echo "$model_to_val_path $model_name $adversarial_method_name"
/data/dans/anaconda3/envs/trl_new/bin/python src/4_edit_and_evaluate_toxic.py \
    --model_to_val_path $model_to_val_path \
    --model_name $model_name \
    --adversarial_method_name $adversarial_method_name \
    --generation_dir eval_results/outputs_$run_id/random_last_layer \
    --metric_dir eval_results/metrics_$run_id/random_last_layer \
    --gpu_id $gpu_id \
    --enhance_neuron_num ${enhance_neuron_num} \
    --enhance_strength ${enhance_strength} \
    --generate_batch_size 4 \
    --suffix_for_output randomll
    # --few_data_for_debug