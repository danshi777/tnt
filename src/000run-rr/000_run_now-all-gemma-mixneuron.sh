gpu_id=$1


adversarial_method_name_array=(
    "DAP"
    "DeepInception"
    "MultiJail"
    # "Jailbroken"
    # "SafeEditWithAdv"
    # "VirtualPersonas"
)
for adversarial_method_name in ${adversarial_method_name_array[@]}; do
    ### ------------------ neuron微调 --------------------
    # 参数各是：gpu_id model_name adversarial_method_name rank run_id
    src/lr2e-5-mixneuron/6_run_sft_and_test_neuron_itm.sh $gpu_id "gemma-2b-it" $adversarial_method_name 8 15
done


# src/000_run_id131415/000_run_now-all-random-randomll.sh $gpu_id


src/00000_run_sft_lora_llama2-7b-chat-22222.sh $gpu_id "run_end"