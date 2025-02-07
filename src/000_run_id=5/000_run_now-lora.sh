gpu_id=$1


### ------------------ lora微调 --------------------
# sft
# 参数各是：gpu_id model_name adversarial_method_name rank run_id
src/2_run_sft_lora_itm.sh $gpu_id "gemma-2b-it" "SafeEditWithAdv" 8 5
# val
# 参数各是：gpu_id model_name adversarial_method_name run_id
src/4_run_test_trained_lora_itm.sh $gpu_id "gemma-2b-it" "SafeEditWithAdv" 5


### ------------------ lora微调 --------------------
# sft
# 参数各是：gpu_id model_name adversarial_method_name rank run_id
src/2_run_sft_lora_itm.sh $gpu_id "llama2-7b-chat" "SafeEditWithAdv" 8 5
# val
# 参数各是：gpu_id model_name adversarial_method_name run_id
src/4_run_test_trained_lora_itm.sh $gpu_id "llama2-7b-chat" "SafeEditWithAdv" 5


# src/00000_run_sft_lora_llama2-7b-chat-22222.sh $gpu_id tttt