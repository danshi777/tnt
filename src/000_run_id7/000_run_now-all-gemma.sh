gpu_id=$1

### ------------------ lora微调 --------------------
# 参数各是：gpu_id model_name adversarial_method_name rank run_id
# src/6_run_sft_and_test_lora_itm.sh $gpu_id "gemma-2b-it" "Jailbroken" 2 7
# src/6_run_sft_and_test_lora_itm.sh $gpu_id "gemma-2b-it" "SafeEditWithAdv" 2 7


### ------------------ neuron 微调 --------------------
# 参数各是：gpu_id model_name adversarial_method_name trainable_param_num run_id
# src/6_run_sft_and_test_neuron_itm.sh $gpu_id "gemma-2b-it" "Jailbroken" 230400 7
# src/6_run_sft_and_test_neuron_itm.sh $gpu_id "gemma-2b-it" "SafeEditWithAdv" 230400 7

# src/000_run_id8/000_run_now-all-gemma.sh $gpu_id

src/000_run_id9/000_run_now-all-gemma.sh $gpu_id

src/00000_run_sft_lora_llama2-7b-chat-22222.sh $gpu_id tttt