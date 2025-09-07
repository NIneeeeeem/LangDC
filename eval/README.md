# Quantitative Evaluation 📊

We provide instructions to evaluate LangDC on  VideoMME, MVBench, LongVideoBench and VSIBench and four open-ended QA Benchmark. Please follow the instructions below,


## VideoMME
The lmms-lab/Video-MME dataset is the comprehensive benchmark designed to evaluate the performance of Multi-modal Large Language Models (MLLMs) in video analysis, encompassing 900 videos across diverse domains, durations, and modalities, with 2,700 human-annotated question-answer pairs. It is introduced in the `Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis` paper. 
Pleae follow the following steps for evaluation,

```bash
# Download and extract MVBench dataset following the official huggingface link
mkdir OpenGVLab
git lfs install
git clone https://huggingface.co/datasets/lmms-lab/Video-MME

# Extract all the videos in OpenGVLab/MVBench/video

# Run inference
bash eval/videomme/test.sh

# Evaluate
python eval/mvbench/evaluation/evaluate_videomme.py
```

## MVBench
MVBench is a comprehensive video understanding benchmark which covers 20 challenging video tasks that cannot be effectively solved with a single frame. It is introduced in the `MVBench: A Comprehensive Multi-modal Video Understanding Benchmark` paper. 
Pleae follow the following steps for evaluation,

```bash
# Download and extract MVBench dataset following the official huggingface link
git lfs install
git clone https://huggingface.co/datasets/OpenGVLab/MVBench

# Extract all the videos in OpenGVLab/MVBench/video

# Run inference
bash eval/mvbench/test.sh

# Evaluate
python eval/mvbench/evaluation/evaluate_mvbench.py
```