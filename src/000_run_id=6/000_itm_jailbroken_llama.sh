gpu_id=$1

### ------------------ lora微调 --------------------
# sft
# 参数各是：gpu_id model_name adversarial_method_name rank run_id
# src/2_run_sft_lora_itm.sh $gpu_id "llama2-7b-chat" "Jailbroken" 8 5
# val
# 参数各是：gpu_id model_name adversarial_method_name run_id
# src/4_run_test_trained_lora_itm.sh $gpu_id "llama2-7b-chat" "Jailbroken" 5


### -------------------- 找神经元 --------------------
# 参数各是：gpu_id model_name adversarial_method_name batch_size(即m) max_new_tokens
src/1_run_analyze_itm.sh $gpu_id "llama2-7b-chat" "Jailbroken" 8 200


### -------------------- neuron微调 --------------------
# sft
# 参数各是：gpu_id model_name adversarial_method_name trainable_param_num run_id
src/2_run_sft_neuron_itm.sh $gpu_id "llama2-7b-chat" "Jailbroken" 4194304 5
# val
# 参数各是：gpu_id model_name adversarial_method_name run_id
src/4_run_test_trained_neuron_itm.sh $gpu_id "llama2-7b-chat" "Jailbroken" 5