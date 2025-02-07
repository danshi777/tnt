gpus=$1

model_path_array=(
    /data1/dans/models/gemma-2b
    # /data1/dans/models/llama2-7b-chat-hf
    # /home/sth/data/code/Meta-Llama-3-8B-Instruct
)
model_name_array=(
    gemma-2b-base
    # llama2-7b-chat
    # llama3-8b-instruct
)

# dev
for ((i=0; i<${#model_path_array[@]}; i++))
do
    model_path=${model_path_array[i]}
    model_name=${model_name_array[i]}
    echo "$model_path $model_name"
    # for enhance_neuron_num in {1..1}
    # do
    #     for enhance_strength in {1..1}
    #     do
    for enhance_neuron_num in {1..15}
    do
        for enhance_strength in {1..16}
        do
            if [[ $enhance_strength -eq 1 && $enhance_neuron_num -gt 1 ]]; then  # skip the case enhance_strength=1 and enhance_neuron_num>1 
                continue
            fi
            # if [[ $enhance_neuron_num -lt 11 && $enhance_strength -lt 19 ]]; then  # skip the case enhance_neuron_num<11 and enhance_strength<19
            #     continue
            # fi
            echo "Evaluating enhanced $model_name with SafeEdit, where enhance_neuron_num=${enhance_neuron_num}, enhance_strength=${enhance_strength}"
            /data/dans/anaconda3/envs/trl_new/bin/python src/4_edit_and_evaluate_toxic.py \
                --model_to_val_path $model_path \
                --data_path data/SafeEdit/train/validation.json \
                --model_name $model_name \
                --dataset_name SafeEdit \
                --neuron_file results/neurons/SafeEdit_gemma-2b-cross_entropy-intermediate-1_neuron.json \
                --output_dir eval_results/dev_results/SafeEdit/outputs_6 \
                --metric_dir eval_results/dev_results/SafeEdit/metrics_6 \
                --gpu_id $gpus \
                --enhance_neuron_num ${enhance_neuron_num} \
                --enhance_strength ${enhance_strength} \
                --method base
                # --few_data_for_debug
        done
    done
done