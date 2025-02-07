gpu_id=$1
model_path_array=(
    /data1/dans/models/gemma-2b
    # /home/sth/data/code/Meta-Llama-3-8B
)
model_name_array=(
    gemma-2b
    # llama3-8b
)
for ((i=0; i<${#model_path_array[@]}; i++))
do
    model_path=${model_path_array[i]}
    model_name=${model_name_array[i]}
    echo "$model_path $model_name"
    /data/dans/anaconda3/envs/trl_new/bin/python src/1_analyze_SafeEdit_fixed_grad.py \
        --model_path $model_path \
        --data_path /data/dans/projects/trl/trl/data/SafeEdit/train/train.json \
        --output_attribution_dir results/attributions/ \
        --output_neuron_dir results/neurons/ \
        --model_name $model_name-cross_entropy-hidden-1 \
        --dataset_name SafeEdit \
        --gpu_id $gpu_id \
        --batch_size 10 \
        --num_batch 1 \
        --batch_size_per_inference 5 \
        --criterion cross_entropy \
        --get_intermediate_size_or_hidden_size hidden
done