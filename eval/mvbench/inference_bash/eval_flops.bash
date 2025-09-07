
GPUS=$1

export PYTHONPATH="./:$PYTHONPATH"

CUDA_VISIBLE_DEVICES=$1 python eval/mvbench/inference/infer_vqa_flops.py \
    --model-path new_weights/dual_3b_mvbench/dual_3b_hidden0_forward \
    --model-base .cache/qwen2_5/3b_instruction \
    --cap-light-llm .cache/qwen2_5/cap_llava_vcap_stage2 \
    --output-dir new_json/debug \
    --is_cap_branch True \
    --is_pool_branch True \
    --hidden_state_layer -9 \
    --debug True