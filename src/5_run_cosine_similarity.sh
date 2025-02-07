gpu_id=$1

export PYTHONPATH="$(pwd)"

CUDA_VISIBLE_DEVICES="${gpu_id}" python src/5_cosine_similarity.py \
    --model_name="gemma-2b-it" \
    --model_id_or_path="/data/cordercorder/pretrained_models/google/gemma-2b-it"