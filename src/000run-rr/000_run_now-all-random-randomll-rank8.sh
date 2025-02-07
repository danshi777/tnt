gpu_id=$1

adversarial_method_name_array=(
    # "VirtualPersonas"
    "DeepInception"
    "Jailbroken"
    "SafeEditWithAdv"
    "DAP"
    "MultiJail"
)
for adversarial_method_name in ${adversarial_method_name_array[@]}; do
    random_seed=60
    ### -------------------- "gemma-2b-it" --------------------
    # 参数各是：gpu_id model_name adversarial_method_name trainable_param_num random_seed run_id
    src/lr2e-5/6_run_sft_and_test_random_itm.sh $gpu_id "gemma-2b-it" $adversarial_method_name 921600 $random_seed 15
    src/lr2e-5/6_run_sft_and_test_random_last_layer_itm.sh $gpu_id "gemma-2b-it" $adversarial_method_name 921600 $random_seed 15

    ### -------------------- "llama2-7b-chat" --------------------
    src/lr2e-5/6_run_sft_and_test_random_itm.sh $gpu_id "llama2-7b-chat" $adversarial_method_name 4194304 $random_seed 15
    src/lr2e-5/6_run_sft_and_test_random_last_layer_itm.sh $gpu_id "llama2-7b-chat" $adversarial_method_name 4194304 $random_seed 15


    random_seed=34
    ### -------------------- "gemma-2b-it" --------------------
    # 参数各是：gpu_id model_name adversarial_method_name trainable_param_num random_seed run_id
    src/lr2e-5/6_run_sft_and_test_random_itm.sh $gpu_id "gemma-2b-it" $adversarial_method_name 921600 $random_seed 15
    src/lr2e-5/6_run_sft_and_test_random_last_layer_itm.sh $gpu_id "gemma-2b-it" $adversarial_method_name 921600 $random_seed 15

    ### -------------------- "llama2-7b-chat" --------------------
    src/lr2e-5/6_run_sft_and_test_random_itm.sh $gpu_id "llama2-7b-chat" $adversarial_method_name 4194304 $random_seed 15
    src/lr2e-5/6_run_sft_and_test_random_last_layer_itm.sh $gpu_id "llama2-7b-chat" $adversarial_method_name 4194304 $random_seed 15


    random_seed=42
    ### -------------------- "gemma-2b-it" --------------------
    # 参数各是：gpu_id model_name adversarial_method_name trainable_param_num random_seed run_id
    src/lr2e-5/6_run_sft_and_test_random_itm.sh $gpu_id "gemma-2b-it" $adversarial_method_name 921600 $random_seed 15
    src/lr2e-5/6_run_sft_and_test_random_last_layer_itm.sh $gpu_id "gemma-2b-it" $adversarial_method_name 921600 $random_seed 15

    ### -------------------- "llama2-7b-chat" --------------------
    src/lr2e-5/6_run_sft_and_test_random_itm.sh $gpu_id "llama2-7b-chat" $adversarial_method_name 4194304 $random_seed 15
    src/lr2e-5/6_run_sft_and_test_random_last_layer_itm.sh $gpu_id "llama2-7b-chat" $adversarial_method_name 4194304 $random_seed 15
done


# ### -------------------- random微调 --------------------
# for i in {1..1}; do
#     while true; do
#         random_seed=$RANDOM
#         let "random_seed %= 100"  # 保证随机数在0-99之间

#         # 检查random_seed是否已经存在于generated_seeds中
#         if [[ ! " ${generated_seeds[@]} " =~ " ${random_seed} " ]]; then
#             generated_seeds+=($random_seed)  # 将随机数加入数组
#             ### -------------------- "gemma-2b-it" "Jailbroken" --------------------
#             # 参数各是：gpu_id model_name adversarial_method_name trainable_param_num random_seed run_id
#             src/lr2e-5/6_run_sft_and_test_random_itm.sh $gpu_id "gemma-2b-it" "Jailbroken" 921600 $random_seed 7
#             src/lr2e-5/6_run_sft_and_test_random_last_layer_itm.sh $gpu_id "gemma-2b-it" "Jailbroken" 921600 $random_seed 7

#             ### -------------------- "gemma-2b-it" "SafeEditWithAdv" --------------------
#             src/lr2e-5/6_run_sft_and_test_random_itm.sh $gpu_id "gemma-2b-it" "SafeEditWithAdv" 921600 $random_seed 7
#             src/lr2e-5/6_run_sft_and_test_random_last_layer_itm.sh $gpu_id "gemma-2b-it" "SafeEditWithAdv" 921600 $random_seed 7

#             ### -------------------- "llama2-7b-chat" "Jailbroken" --------------------
#             src/lr2e-5/6_run_sft_and_test_random_itm.sh $gpu_id "llama2-7b-chat" "Jailbroken" 4194304 $random_seed 7
#             src/lr2e-5/6_run_sft_and_test_random_last_layer_itm.sh $gpu_id "llama2-7b-chat" "Jailbroken" 4194304 $random_seed 7

#             ### -------------------- "llama2-7b-chat" "SafeEditWithAdv" --------------------
#             src/lr2e-5/6_run_sft_and_test_random_itm.sh $gpu_id "llama2-7b-chat" "SafeEditWithAdv" 4194304 $random_seed 7
#             src/lr2e-5/6_run_sft_and_test_random_last_layer_itm.sh $gpu_id "llama2-7b-chat" "SafeEditWithAdv" 4194304 $random_seed 7
#             break  # 跳出循环
#         fi
#     done
# done

# src/00000_run_sft_lora_llama2-7b-chat-22222.sh $gpu_id tttt