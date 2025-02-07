gpu_id=$1


adversarial_method_name_array=(
    "DAP"
    "DeepInception"
    "Jailbroken"
    # "SafeEditWithAdv"
    # "VirtualPersonas"
)
for adversarial_method_name in ${adversarial_method_name_array[@]}; do
    ### -------------------- 找神经元 --------------------
    # 参数各是：gpu_id model_name adversarial_method_name batch_size(即m) max_new_tokens top_n
    src/1_run_analyze_itm.sh $gpu_id "llama2-7b-chat" $adversarial_method_name 8 200 100


    ### ------------------ lora微调 --------------------
    # 参数各是：gpu_id model_name adversarial_method_name rank run_id
    src/lr2e-5/6_run_sft_and_test_lora_itm.sh $gpu_id "llama2-7b-chat" $adversarial_method_name 2 13

    ### ------------------ neuron 微调 --------------------
    # 参数各是：gpu_id model_name adversarial_method_name trainable_param_num run_id
    src/lr2e-5/6_run_sft_and_test_neuron_itm.sh $gpu_id "llama2-7b-chat" $adversarial_method_name 1048576 13


    ### ------------------ lora微调 --------------------
    # 参数各是：gpu_id model_name adversarial_method_name rank run_id
    src/lr2e-5/6_run_sft_and_test_lora_itm.sh $gpu_id "llama2-7b-chat" $adversarial_method_name 4 14

    ### ------------------ neuron 微调 --------------------
    # 参数各是：gpu_id model_name adversarial_method_name trainable_param_num run_id
    src/lr2e-5/6_run_sft_and_test_neuron_itm.sh $gpu_id "llama2-7b-chat" $adversarial_method_name 2097152 14


    ### ------------------ lora微调 --------------------
    # 参数各是：gpu_id model_name adversarial_method_name rank run_id
    src/lr2e-5/6_run_sft_and_test_lora_itm.sh $gpu_id "llama2-7b-chat" $adversarial_method_name 8 15

    ### ------------------ neuron 微调 --------------------
    # 参数各是：gpu_id model_name adversarial_method_name trainable_param_num run_id
    src/lr2e-5/6_run_sft_and_test_neuron_itm.sh $gpu_id "llama2-7b-chat" $adversarial_method_name 4194304 15
done


src/00000_run_sft_lora_llama2-7b-chat-22222.sh $gpu_id tttt