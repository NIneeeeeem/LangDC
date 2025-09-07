export PYTHONPATH="./:$PYTHONPATH"
export CUDA_LAUNCH_BLOCKING=1
# export DATASET_DIR=playground/data

CUDA_VISIBLE_DEVICES=$1 python eval/longvideobench/inference/infer.py \
    --model-path weights/mvbench_weights/dual_branch_3b \
    --model-base .cache/qwen2_5/3b_instruction \
    --cap-light-llm .cache/qwen2_5/cap_llava_vcap_stage2_0210 \
    --output-dir weights/mvbench_weights/dual_branch_3b/longbench_results4_16_with \
    --is_cap_branch True \
    --is_pool_branch True \
    --hidden_state_layer -9 \
    --conv-mode qwen2 \