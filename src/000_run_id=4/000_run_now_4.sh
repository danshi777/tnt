gpu_id=$1

### -------------------- neuron微调 --------------------
# sft
# 参数各是：gpu_id model_name adversarial_method_name trainable_param_num run_id
src/2_run_sft_neuron_itm.sh $gpu_id "llama2-7b-chat" "SafeEditWithAdv" 1048576 4
# val
# 参数各是：gpu_id model_name adversarial_method_name run_id
src/4_run_test_trained_neuron_itm.sh $gpu_id "llama2-7b-chat" "SafeEditWithAdv" 4

src/00000_run_sft_lora_llama2-7b-chat-22222.sh $gpu_id tttt