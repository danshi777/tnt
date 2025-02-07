gpu_id=$1

adversarial_method_name_array=(
    # "VirtualPersonas"
    "DeepInception"
    "DAP"
    "Jailbroken"
    "SafeEditWithAdv"
)
for adversarial_method_name in ${adversarial_method_name_array[@]}; do
    ### -------------------- 找神经元 --------------------
    # 参数各是：gpu_id model_name adversarial_method_name batch_size(即m) max_new_tokens top_n
    # src/1_run_analyze_itm.sh $gpu_id "llama2-7b-chat" $adversarial_method_name 8 200 50


    ### ------------------ lora微调 --------------------
    # 参数各是：gpu_id model_name adversarial_method_name rank run_id
    src/lr5e-5/6_run_sft_and_test_lora_itm.sh $gpu_id "llama2-7b-chat" $adversarial_method_name 2 10

    ### ------------------ neuron 微调 --------------------
    # 参数各是：gpu_id model_name adversarial_method_name trainable_param_num run_id
    src/lr5e-5/6_run_sft_and_test_neuron_itm.sh $gpu_id "llama2-7b-chat" $adversarial_method_name 1048576 10


    ### ------------------ lora微调 --------------------
    # 参数各是：gpu_id model_name adversarial_method_name rank run_id
    src/lr5e-5/6_run_sft_and_test_lora_itm.sh $gpu_id "llama2-7b-chat" $adversarial_method_name 4 11

    ### ------------------ neuron 微调 --------------------
    # 参数各是：gpu_id model_name adversarial_method_name trainable_param_num run_id
    src/lr5e-5/6_run_sft_and_test_neuron_itm.sh $gpu_id "llama2-7b-chat" $adversarial_method_name 2097152 11


    ### -------------------- 找神经元 --------------------
    # 参数各是：gpu_id model_name adversarial_method_name batch_size(即m) max_new_tokens top_n
    src/1_run_analyze_itm.sh $gpu_id "llama2-7b-chat" $adversarial_method_name 8 200 100


    ### ------------------ lora微调 --------------------
    # 参数各是：gpu_id model_name adversarial_method_name rank run_id
    src/lr5e-5/6_run_sft_and_test_lora_itm.sh $gpu_id "llama2-7b-chat" $adversarial_method_name 8 12

    ### ------------------ neuron 微调 --------------------
    # 参数各是：gpu_id model_name adversarial_method_name trainable_param_num run_id
    src/lr5e-5/6_run_sft_and_test_neuron_itm.sh $gpu_id "llama2-7b-chat" $adversarial_method_name 4194304 12
done

src/00000_run_sft_lora_llama2-7b-chat-22222.sh $gpu_id tttt
# src/00000_run_sft_lora_llama2-7b-chat-22222.sh 0 tttt