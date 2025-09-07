import os

# DATASET_DIR = os.environ.get("DATASET_DIR", "playground/data")
DATASET_DIR = os.environ.get("DATASET_DIR", ".cache/instruction_data")
PRETRAIN_DATASET_DIR = os.environ.get("PRETRAIN_DATASET_DIR", ".cache/instruction_data")

CC3M_595K = {
    "annotation_path": f"{PRETRAIN_DATASET_DIR}/pretraining/CC3M-595K/chat.json",
    "data_path": f"{PRETRAIN_DATASET_DIR}/pretraining/CC3M-595K",
}

COCO_CAP = {
    "annotation_path": f"{PRETRAIN_DATASET_DIR}/pretraining/COCO/coco_cap_chat.json",
    "data_path": f"{PRETRAIN_DATASET_DIR}/pretraining/COCO/train2014",
}

COCO_REG = {
    "annotation_path": f"{PRETRAIN_DATASET_DIR}/pretraining/COCO/coco_reg_chat.json",
    "data_path": f"{PRETRAIN_DATASET_DIR}/pretraining/COCO/train2014",
}

COCO_REC = {
    "annotation_path": f"{PRETRAIN_DATASET_DIR}/pretraining/COCO/coco_rec_chat.json",
    "data_path": f"{PRETRAIN_DATASET_DIR}/pretraining/COCO/train2014",
}

CONV_VideoChatGPT = {
    "annotation_path": f"{DATASET_DIR}/annotations/conversation_videochatgpt.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/Activity_Videos",
}

VCG_HUMAN = {
    "annotation_path": f"{DATASET_DIR}/annotations/vcg_human_annotated.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/Activity_Videos",
}

VCG_PLUS_112K = {
    "annotation_path": f"{DATASET_DIR}/annotations/vcg-plus_112K.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/Activity_Videos",
}

CAPTION_VIDEOCHAT = {
    "annotation_path": f"{DATASET_DIR}/annotations/caption_videochat.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/webvid",
}

CAPTION_PANDA2M = {
    "annotation_path": f"{DATASET_DIR}/annotations/caption_panda2m_filter.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/panda2m_filter",
}

CLASSIFICATION_K710 = {
    "annotation_path": f"{DATASET_DIR}/annotations/classification_k710.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/k710",
}
CLASSIFICATION_K600_Cap = {
    "annotation_path": f"{DATASET_DIR}/annotations/k600_data.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/k710",
}
CLASSIFICATION_K600_Cap_nltk = {
    "annotation_path": f"{DATASET_DIR}/annotations/k600_nltk_data.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/k710",
}
CLASSIFICATION_K600_Cap_vn = {
    "annotation_path": f"{DATASET_DIR}/annotations/k600_vn_data.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/k710",
}
CLASSIFICATION_K600_Cap_debug = {
    "annotation_path": f"{DATASET_DIR}/annotations/k600_debug_data.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/k710",
}

CLASSIFICATION_SSV2 = {
    "annotation_path": f"{DATASET_DIR}/annotations/classification_ssv2.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/ssv2",
}

CONV_VideoChat1 = {
    "annotation_path": f"{DATASET_DIR}/annotations/conversation_videochat1.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/videochat_it",
}

REASONING_NExTQA = {
    "annotation_path": f"{DATASET_DIR}/annotations/reasoning_next_qa.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/NExTQA",
}

REASONING_CLEVRER_QA = {
    "annotation_path": f"{DATASET_DIR}/annotations/reasoning_clevrer_qa.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/clevrer",
}

REASONING_CLEVRER_MC = {
    "annotation_path": f"{DATASET_DIR}/annotations/reasoning_clevrer_mc.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/clevrer",
}

VQA_WEBVID_QA = {
    "annotation_path": f"{DATASET_DIR}/annotations/vqa_webvid_qa.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/webvid",
}

VQA_DEBUG = {
    "annotation_path": f"{DATASET_DIR}/llavaov_it/reasoning_next_qa.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/NExTQA",
}
# ===================== here for vqa cap dataset ========================
REASONING_NExTQA_vqa = {
    "annotation_path": f"{DATASET_DIR}/llavaov_it/reasoning_next_qa.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/NExTQA",
}
REASONING_CLEVRER_QA_vqa = {
    "annotation_path": f"{DATASET_DIR}/llavaov_it/reasoning_clevrer_qa.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/clevrer",
}
REASONING_CLEVRER_MC_vqa = {
    "annotation_path": f"{DATASET_DIR}/llavaov_it/reasoning_clevrer_mc.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/clevrer",
}
VQA_WEBVID_QA_vqa = {
    "annotation_path": f"{DATASET_DIR}/llavaov_it/vqa_webvid_qa.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/webvid",
}
CLASSIFICATION_K710_vqa = {
    "annotation_path": f"{DATASET_DIR}/llavaov_it/classification_k710.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/k710",
}
CLASSIFICATION_SSV2_vqa = {
    "annotation_path": f"{DATASET_DIR}/llavaov_it/classification_ssv2.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/ssv2",
}
CONV_VideoChatGPT_vqa = {
    "annotation_path": f"{DATASET_DIR}/llavaov_it/conversation_videochatgpt.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/Activity_Videos",
}


VCG_HUMAN_vqa = {
    "annotation_path": f"{DATASET_DIR}/llavaov_it/vcg_human_annotated.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/Activity_Videos",
}

VCG_PLUS_112K_vqa = {
    "annotation_path": f"{DATASET_DIR}/llavaov_it/vcg-plus_112K.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/Activity_Videos",
}

CAPTION_VIDEOCHAT_vqa = {
    "annotation_path": f"{DATASET_DIR}/llavaov_it/caption_videochat.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/webvid",
}
VQA_WEBVID_QA_vqa = {
    "annotation_path": f"{DATASET_DIR}/llavaov_it/vqa_webvid_qa.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/webvid",
}
CONV_VideoChat1_vqa = {
    "annotation_path": f"{DATASET_DIR}/llavaov_it/conversation_videochat1.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/videochat_it",
}
# ===================== here for vqa cap sub dataset ========================
REASONING_NExTQA_vqa_sub = {
    "annotation_path": f"{DATASET_DIR}/merged_cap_vqa_ann/reasoning_next_qa.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/NExTQA",
}
REASONING_CLEVRER_QA_vqa_sub = {
    "annotation_path": f"{DATASET_DIR}/merged_cap_vqa_ann/reasoning_clevrer_qa.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/clevrer",
}
REASONING_CLEVRER_MC_vqa_sub = {
    "annotation_path": f"{DATASET_DIR}/merged_cap_vqa_ann/reasoning_clevrer_mc.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/clevrer",
}
VQA_WEBVID_QA_vqa_sub = {
    "annotation_path": f"{DATASET_DIR}/merged_cap_vqa_ann/vqa_webvid_qa.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/webvid",
}
CLASSIFICATION_K710_vqa_sub = {
    "annotation_path": f"{DATASET_DIR}/merged_cap_vqa_ann/classification_k710.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/k710",
}
CLASSIFICATION_SSV2_vqa_sub = {
    "annotation_path": f"{DATASET_DIR}/merged_cap_vqa_ann/classification_ssv2.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/ssv2",
}
CONV_VideoChatGPT_vqa_sub = {
    "annotation_path": f"{DATASET_DIR}/merged_cap_vqa_ann/conversation_videochatgpt.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/Activity_Videos",
}

# ===================== here for tarsier vqa cap dataset ========================
REASONING_NExTQA_vqa_tarsier = {
    "annotation_path": f"{DATASET_DIR}/tarsier_it/reasoning_next_qa.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/NExTQA",
}
REASONING_CLEVRER_QA_vqa_tarsier = {
    "annotation_path": f"{DATASET_DIR}/tarsier_it/reasoning_clevrer_qa.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/clevrer",
}
REASONING_CLEVRER_MC_vqa_tarsier = {
    "annotation_path": f"{DATASET_DIR}/tarsier_it/reasoning_clevrer_mc.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/clevrer",
}
VQA_WEBVID_QA_vqa_tarsier = {
    "annotation_path": f"{DATASET_DIR}/tarsier_it/vqa_webvid_qa.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/webvid",
}
CLASSIFICATION_K710_vqa_tarsier = {
    "annotation_path": f"{DATASET_DIR}/tarsier_it/classification_k710.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/k710",
}
CLASSIFICATION_SSV2_vqa_tarsier = {
    "annotation_path": f"{DATASET_DIR}/tarsier_it/classification_ssv2.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/ssv2",
}
CONV_VideoChatGPT_vqa_tarsier = {
    "annotation_path": f"{DATASET_DIR}/tarsier_it/conversation_videochatgpt.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/Activity_Videos",
}
CONV_VideoChat1_vqa_tarsier = {
    "annotation_path": f"{DATASET_DIR}/tarsier_it/conversation_videochat1.json",
    "data_path": f"{DATASET_DIR}/instruction_tuning/videochat_it",
}

### image caption
CAP_LLAVARECAP = {
    "annotation_path": f"{PRETRAIN_DATASET_DIR}/LLaVA-ReCap-558K/data",
}