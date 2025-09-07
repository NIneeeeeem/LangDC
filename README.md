
<div align="center">
  <h1>Seeing More, Saying More: Lightweight Language Experts are Dynamic Video Token Compressors</h1> 
</div>

<h5 align="center"> 

[![arXiv](https://img.shields.io/badge/LangDC-2509.00969-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2509.00969)
[![GitHub](https://img.shields.io/badge/GitHub-Code-green?logo=github)](https://github.com/NIneeeeeem/LangDC)
[![HuggingFace](https://img.shields.io/badge/🤗%20Model-Huggingface-yellow)](https://huggingface.co/Wangxc1000/LangDC)

 <br>

</h5>

## LangDC Overview 

Current large video-language models face efficiency issues due to processing massive visual tokens. Existing fixed-ratio token compression ignores varying semantic density across video clips. Consequently, this lead to inadequate representation of information-rich clips due to insufficient tokens and unnecessary computation on static or content-poor ones. To address this, we propose LangDC, a Language-aware Dynamic Token Compressor. LangDC leverages a lightweight language model to describe video clips, converting them into soft caption tokens as visual representations. Trained with our proposed semantic density-aware supervision, LangDC aims to 1) cover key visual cues necessary for downstream task reasoning and 2) dynamically adjust compression ratios based on scene richness, reflected by descriptions length.

<p align="center">
  <img src="asset/motivation_comparision.png" alt="Comparison of LangDC and existing token compressors.">
</p>

## Contributions 
1) We propose LangDC, a novel language-aware token compression strategy. Using soft language tokens for visual representation, it adaptively adjusts compression ratios, improving token utilization over fixed-ratio techniques. 

2) We propose semantic density-aware supervision for the token compressors. By explicitly providing reconstruction targets for token compression, we enable the derivation of a more compact feature set that is not only aware of information richness but also preserves key visual cues. 

3) Experimental results demonstrate that our method reduces FLOPs by 49\% relative to the strong baseline VideoGPT+, while maintaining competitive performance. Additional qualitative results show adaptive compression based on video clip semantic density.

<p align="center">
  <img src="asset/fig_method.png" alt="Overview of the LangDC.">
</p>

## Installation

We recommend setting up a conda environment for the project:
```shell
conda create --name=langdc python=3.11
conda activate langdc

git clone https://github.com/NIneeeeeem/LangDC.git
cd LangDC

pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.41.0

pip install -r requirements.txt

export PYTHONPATH="./:$PYTHONPATH"
```
Additionally, install [FlashAttention](https://github.com/HazyResearch/flash-attention) for training,
```shell
pip install ninja

git clone https://github.com/HazyResearch/flash-attention.git
cd flash-attention
python setup.py install
```
---

## Quantitative Evaluation 📊
We provide instructions to reproduce LangDC results on VideoMME, MVBench, LongVideoBench, VSIBench and four open-ended QA Benchmark. Please follow the instructions at [eval/README.md](eval/README.md).

To reproduce the results in Table 1 of the Motivation chapter, please refer to [this repository](https://github.com/NIneeeeeem/VideoGPT-tokenadapter.git).


## Citations 📜:

If you're using LangDC in your research or applications, please give us a star ⭐ to support us and cite using this BibTeX:
```bibtex
@misc{wang2025seeing,
    title={Seeing More, Saying More: Lightweight Language Experts are Dynamic Video Token Compressors},
    author={Xiangchen Wang and Jinrui Zhang and Teng Wang and Haigang Zhang and Feng Zheng},
    year={2025},
    eprint={2509.00969},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Acknowledgements 

+ [Video-ChatGPT+](https://github.com/mbzuai-oryx/VideoGPT-plus): A pioneering attempt in Video-based conversation models.
+ [LLaVA](https://github.com/haotian-liu/LLaVA): Our code base is build upon LLaVA and Video-ChatGPT+.