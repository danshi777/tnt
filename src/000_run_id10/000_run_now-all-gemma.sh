gpu_id=$1

### ------------------ lora微调 --------------------
# 参数各是：gpu_id model_name adversarial_method_name rank run_id
src/lr5e-5/6_run_sft_and_test_lora_itm.sh $gpu_id "gemma-2b-it" "Jailbroken" 2 10
src/lr5e-5/6_run_sft_and_test_lora_itm.sh $gpu_id "gemma-2b-it" "SafeEditWithAdv" 2 10


### ------------------ neuron 微调 --------------------
# 参数各是：gpu_id model_name adversarial_method_name trainable_param_num run_id
src/lr5e-5/6_run_sft_and_test_neuron_itm.sh $gpu_id "gemma-2b-it" "Jailbroken" 230400 10
src/lr5e-5/6_run_sft_and_test_neuron_itm.sh $gpu_id "gemma-2b-it" "SafeEditWithAdv" 230400 10


### ------------------ lora微调 --------------------
# 参数各是：gpu_id model_name adversarial_method_name rank run_id
src/lr5e-5/6_run_sft_and_test_lora_itm.sh $gpu_id "gemma-2b-it" "Jailbroken" 4 11
src/lr5e-5/6_run_sft_and_test_lora_itm.sh $gpu_id "gemma-2b-it" "SafeEditWithAdv" 4 11


### ------------------ neuron 微调 --------------------
# 参数各是：gpu_id model_name adversarial_method_name trainable_param_num run_id
src/lr5e-5/6_run_sft_and_test_neuron_itm.sh $gpu_id "gemma-2b-it" "Jailbroken" 460800 11
src/lr5e-5/6_run_sft_and_test_neuron_itm.sh $gpu_id "gemma-2b-it" "SafeEditWithAdv" 460800 11


### ------------------ lora微调 --------------------
# 参数各是：gpu_id model_name adversarial_method_name rank run_id
src/lr5e-5/6_run_sft_and_test_lora_itm.sh $gpu_id "gemma-2b-it" "Jailbroken" 8 12
src/lr5e-5/6_run_sft_and_test_lora_itm.sh $gpu_id "gemma-2b-it" "SafeEditWithAdv" 8 12


### ------------------ neuron 微调 --------------------
# 参数各是：gpu_id model_name adversarial_method_name trainable_param_num run_id
src/lr5e-5/6_run_sft_and_test_neuron_itm.sh $gpu_id "gemma-2b-it" "Jailbroken" 921600 12
src/lr5e-5/6_run_sft_and_test_neuron_itm.sh $gpu_id "gemma-2b-it" "SafeEditWithAdv" 921600 12


src/00000_run_sft_lora_llama2-7b-chat-22222.sh $gpu_id tttt