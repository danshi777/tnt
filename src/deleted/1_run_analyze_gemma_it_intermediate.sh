gpu_id=$1
adversarial_method_name_array=(
    # SafeEditWithAdv
    Jailbroken
    # PAP
)
model_name=gemma-2b-it
for adversarial_method_name in ${adversarial_method_name_array[@]};
do
    echo "$model_path $model_name $adversarial_method_name"
    /data/dans/anaconda3/envs/trl_new/bin/python src/1_analyze_SafeEdit_fixed_grad.py \
        --model_name $model_name-cross_entropy-intermediate-1 \
        --adversarial_method_name $adversarial_method_name \
        --output_attribution_dir results/attributions/ \
        --output_neuron_dir results/neurons \
        --gpu_id $gpu_id \
        --num_of_examples 100 \
        --batch_size 8 \
        --num_batch 1 \
        --batch_size_per_inference 1 \
        --max_new_tokens 200 \
        --criterion cross_entropy \
        --get_intermediate_size_or_hidden_size intermediate \
        --threshold 0.01 \
        --top_n 50
        # --only_process_neurons
done