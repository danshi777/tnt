gpu_id=$1
enhance_neuron_num=0
enhance_strength=1

# gemma-2b-it-neuron
model_to_val_path=saved_models/gemma-2b-it/hd/sft_neuron_Jailbroken_2
model_name=gemma-2b-it-neuron
adversarial_method_name=Jailbroken
echo "$model_to_val_path $model_name $adversarial_method_name"
# echo "Evaluating enhanced $model_name with SafeEdit, where enhance_neuron_num=${enhance_neuron_num}, enhance_strength=${enhance_strength}"
/data/dans/anaconda3/envs/trl_new/bin/python src/4_edit_and_evaluate_toxic.py \
    --model_to_val_path $model_to_val_path \
    --model_name $model_name \
    --adversarial_method_name $adversarial_method_name \
    --output_dir eval_results/outputs_2 \
    --metric_dir eval_results/metrics_2 \
    --gpu_id $gpu_id \
    --enhance_neuron_num ${enhance_neuron_num} \
    --enhance_strength ${enhance_strength} \
    --batch_size 4
    # --few_data_for_debug