export PYTHONPATH="$(pwd)"

/data/dans/anaconda3/envs/trl_new/bin/python src/7_plot_neuron_distribution.py \
    --model_name gemma-2b \
    --num_layers 18 \
    --neuron_file results/neurons/SafeEdit_gemma-2b-cross_entropy-intermediate-1_neuron.json \
    --neuron_num 113


# /data/dans/anaconda3/envs/kns/bin/python src/7_plot_kn_llama2-7b.py \
#     --model_name bert \
#     --kn_file_path results/kn/Memo-bert-kn_bag-context.json \
#     --num_layers 12


# /data/dans/anaconda3/envs/kns/bin/python src/7_plot_kn_distribution_models.py \
#     --model_name lama2-7b \
#     --kn_file_path results/kn/Memo-llama2_7b-kn_bag-context.json \
#     --num_layers 32


# /data/dans/anaconda3/envs/kns/bin/python src/7_plot_kn_distribution_models.py \
#     --model_name llama3-8b \
#     --kn_file_path results/kn/Memo-llama3-8b-kn_bag-context.json \
#     --num_layers 32

# /data/dans/anaconda3/envs/kns/bin/python src/7_plot_kn_distribution_models.py \
#     --model_name lama2-13b \
#     --kn_file_path results/kn/Memo-llama2_13b-kn_bag-context.json \
#     --num_layers 40


# /data/dans/anaconda3/envs/kns/bin/python src/7_plot_kn_distribution_Amber.py \
#     --model_name Amber-39 \
#     --kn_file_path results/kn/Memo-Amber-39-kn_bag-context.json \
#     --num_layers 32


# /data/dans/anaconda3/envs/kns/bin/python src/7_plot_kn_distribution_Amber.py \
#     --model_name Amber-119 \
#     --kn_file_path results/kn/Memo-Amber-119-kn_bag-context.json \
#     --num_layers 32


# /data/dans/anaconda3/envs/kns/bin/python src/7_plot_kn_distribution_Amber.py \
#     --model_name Amber-199 \
#     --kn_file_path results/kn/Memo-Amber-199-kn_bag-context.json \
#     --num_layers 32

# /data/dans/anaconda3/envs/kns/bin/python src/7_plot_kn_distribution_Amber.py \
#     --model_name Amber-279 \
#     --kn_file_path results/kn/Memo-Amber-279-kn_bag-context.json \
#     --num_layers 32

# /data/dans/anaconda3/envs/kns/bin/python src/7_plot_kn_distribution_Amber.py \
#     --model_name Amber-359 \
#     --kn_file_path results/kn/Memo-Amber-359-kn_bag-context.json \
#     --num_layers 32