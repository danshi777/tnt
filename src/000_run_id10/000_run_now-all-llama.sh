gpu_id=$1

### ------------------ lora微调 --------------------
# 参数各是：gpu_id model_name adversarial_method_name rank run_id
src/lr5e-5/6_run_sft_and_test_lora_itm.sh $gpu_id "llama2-7b-chat" "Jailbroken" 2 10
src/lr5e-5/6_run_sft_and_test_lora_itm.sh $gpu_id "llama2-7b-chat" "SafeEditWithAdv" 2 10


### ------------------ neuron 微调 --------------------
# 参数各是：gpu_id model_name adversarial_method_name trainable_param_num run_id
src/lr5e-5/6_run_sft_and_test_neuron_itm.sh $gpu_id "llama2-7b-chat" "Jailbroken" 1048576 10
src/lr5e-5/6_run_sft_and_test_neuron_itm.sh $gpu_id "llama2-7b-chat" "SafeEditWithAdv" 1048576 10


### ------------------ lora微调 --------------------
# 参数各是：gpu_id model_name adversarial_method_name rank run_id
src/lr5e-5/6_run_sft_and_test_lora_itm.sh $gpu_id "llama2-7b-chat" "Jailbroken" 4 11
src/lr5e-5/6_run_sft_and_test_lora_itm.sh $gpu_id "llama2-7b-chat" "SafeEditWithAdv" 4 11


### ------------------ neuron 微调 --------------------
# 参数各是：gpu_id model_name adversarial_method_name trainable_param_num run_id
src/lr5e-5/6_run_sft_and_test_neuron_itm.sh $gpu_id "llama2-7b-chat" "Jailbroken" 2097152 11
src/lr5e-5/6_run_sft_and_test_neuron_itm.sh $gpu_id "llama2-7b-chat" "SafeEditWithAdv" 2097152 11


### ------------------ lora微调 --------------------
# 参数各是：gpu_id model_name adversarial_method_name rank run_id
src/lr5e-5/6_run_sft_and_test_lora_itm.sh $gpu_id "llama2-7b-chat" "Jailbroken" 8 12
src/lr5e-5/6_run_sft_and_test_lora_itm.sh $gpu_id "llama2-7b-chat" "SafeEditWithAdv" 8 12


### ------------------ neuron 微调 --------------------
# 参数各是：gpu_id model_name adversarial_method_name trainable_param_num run_id
src/lr5e-5/6_run_sft_and_test_neuron_itm.sh $gpu_id "llama2-7b-chat" "Jailbroken" 4194304 12
src/lr5e-5/6_run_sft_and_test_neuron_itm.sh $gpu_id "llama2-7b-chat" "SafeEditWithAdv" 4194304 12


src/00000_run_sft_lora_llama2-7b-chat-22222.sh $gpu_id tttt