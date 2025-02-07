gpu_id=$1
model_name=$2
adversarial_method_name=$3
    # SafeEditWithAdv
    # Jailbroken
    # PAP
batch_size=$4
max_new_tokens=$5
top_n=$6
get_intermediate_size_or_hidden_size="intermediate"
    # intermediate
    # hidden

export PYTHONPATH="$(pwd)"

echo "model_name=$model_name, adversarial_method_name=$adversarial_method_name, batch_size=$batch_size, max_new_tokens=$max_new_tokens, top_n=$top_n, $get_intermediate_size_or_hidden_size"
/data/dans/anaconda3/envs/trl_new/bin/python src/1_analyze_SafeEdit_fixed_grad.py \
    --model_name $model_name \
    --adversarial_method_name $adversarial_method_name \
    --output_attribution_dir results/attributions/ \
    --output_neuron_dir results/neurons \
    --gpu_id $gpu_id \
    --num_of_examples 100 \
    --batch_size $batch_size \
    --num_batch 1 \
    --batch_size_per_inference 1 \
    --max_new_tokens $max_new_tokens \
    --criterion cross_entropy \
    --get_intermediate_size_or_hidden_size $get_intermediate_size_or_hidden_size \
    --threshold 0.01 \
    --top_n $top_n
    # --only_process_neurons
