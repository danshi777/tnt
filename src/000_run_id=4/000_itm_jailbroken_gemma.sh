gpu_id=$1

### ------------------ lora微调 --------------------
# sft
# 参数各是：gpu_id model_name adversarial_method_name run_id
src/2_run_sft_lora_itm.sh $gpu_id "gemma-2b-it" "Jailbroken" 4
# val
# 参数各是：gpu_id model_name adversarial_method_name run_id
src/4_run_test_trained_lora_itm.sh $gpu_id "gemma-2b-it" "Jailbroken" 4


### -------------------- 找神经元 --------------------
# 参数各是：gpu_id model_name adversarial_method_name batch_size(即m) max_new_tokens
# src/1_run_analyze_itm.sh $gpu_id "gemma-2b-it" "Jailbroken" 8 200


### -------------------- neuron微调 --------------------
# sft
# 参数各是：gpu_id model_name adversarial_method_name trainable_param_num run_id
# src/2_run_sft_neuron_itm.sh $gpu_id "gemma-2b-it" "Jailbroken" 230400 4
# val
# 参数各是：gpu_id model_name adversarial_method_name run_id
# src/4_run_test_trained_neuron_itm.sh $gpu_id "gemma-2b-it" "Jailbroken" 4