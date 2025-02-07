export PYTHONPATH="${workspaceFolder}"
# accelerate launch src/dpo_llama2.py \
CUDA_VISIBLE_DEVICES="2" python src/3_dpo_llama2.py \
    --model_name_or_path="/data/dans/projects/trl/trl/saved_models/sft" \
    --data_dir="/data/dans/projects/trl/trl/data/SafeEdit/train/" \
    --run_name="dpo_llama2-7b_safeedit" \
    --output_dir="./saved_models/dpo" \
    --use_peft \
    --lora_r=8 \
    --lora_alpha=8 \
    --lora_dropout=0.0 \
    --per_device_train_batch_size=2 \
    --optim="rmsprop_bnb_32bit"