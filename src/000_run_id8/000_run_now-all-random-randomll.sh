gpu_id=$1

### -------------------- random微调 --------------------
for i in {1..4}; do
    while true; do
        random_seed=$RANDOM
        let "random_seed %= 100"  # 保证随机数在0-99之间

        # 检查random_seed是否已经存在于generated_seeds中
        if [[ ! " ${generated_seeds[@]} " =~ " ${random_seed} " ]]; then
            generated_seeds+=($random_seed)  # 将随机数加入数组
            ### -------------------- "gemma-2b-it" "Jailbroken" --------------------
            # 参数各是：gpu_id model_name adversarial_method_name trainable_param_num random_seed run_id
            src/6_run_sft_and_test_random_itm.sh $gpu_id "gemma-2b-it" "Jailbroken" 460800 $random_seed 8
            src/6_run_sft_and_test_random_last_layer_itm.sh $gpu_id "gemma-2b-it" "Jailbroken" 460800 $random_seed 8

            ### -------------------- "gemma-2b-it" "SafeEditWithAdv" --------------------
            src/6_run_sft_and_test_random_itm.sh $gpu_id "gemma-2b-it" "SafeEditWithAdv" 460800 $random_seed 8
            src/6_run_sft_and_test_random_last_layer_itm.sh $gpu_id "gemma-2b-it" "SafeEditWithAdv" 460800 $random_seed 8

            ### -------------------- "llama2-7b-chat" "Jailbroken" --------------------
            src/6_run_sft_and_test_random_itm.sh $gpu_id "llama2-7b-chat" "Jailbroken" 2097152 $random_seed 8
            src/6_run_sft_and_test_random_last_layer_itm.sh $gpu_id "llama2-7b-chat" "Jailbroken" 2097152 $random_seed 8

            ### -------------------- "llama2-7b-chat" "SafeEditWithAdv" --------------------
            src/6_run_sft_and_test_random_itm.sh $gpu_id "llama2-7b-chat" "SafeEditWithAdv" 2097152 $random_seed 8
            src/6_run_sft_and_test_random_last_layer_itm.sh $gpu_id "llama2-7b-chat" "SafeEditWithAdv" 2097152 $random_seed 8
            break  # 跳出循环
        fi
    done
done