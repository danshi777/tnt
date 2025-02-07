gpu_id=$1

### ------------------ lora微调 --------------------
# 参数各是：gpu_id model_name adversarial_method_name rank run_id
src/6_run_sft_and_test_lora_itm.sh $gpu_id "llama2-7b-chat" "Jailbroken" 8 9
src/6_run_sft_and_test_lora_itm.sh $gpu_id "llama2-7b-chat" "SafeEditWithAdv" 8 9


### ------------------ neuron 微调 --------------------
# 参数各是：gpu_id model_name adversarial_method_name trainable_param_num run_id
src/6_run_sft_and_test_neuron_itm.sh $gpu_id "llama2-7b-chat" "Jailbroken" 4194304 9
src/6_run_sft_and_test_neuron_itm.sh $gpu_id "llama2-7b-chat" "SafeEditWithAdv" 4194304 9


# src/00000_run_sft_lora_llama2-7b-chat-22222.sh $gpu_id tttt