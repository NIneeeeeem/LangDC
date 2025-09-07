import torch.nn as nn
import re, torch
from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2Model
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from typing import List, Optional, Tuple, Union
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from functools import partial
from timm.layers.norm_act import LayerNormAct2d
from torchvision.ops.misc import SqueezeExcitation as SElayer
from torchvision.models.mobilenetv3 import InvertedResidual, InvertedResidualConfig
import math
import torch

class LDPBlock(nn.Module):
    # Lightweight Downsample Projector Block

    def __init__(self, config=None):
        super().__init__()

        inc, ouc = config.mm_hidden_size, config.hidden_size
        layer_norm = partial(LayerNormAct2d, act_layer=None)
        se_layer = partial(SElayer, scale_activation=nn.Hardsigmoid)
        self.mlp = nn.Sequential(
            nn.Identity(), nn.Linear(inc, ouc), nn.GELU(), nn.Linear(ouc, ouc)
        )
        self.mb_block = nn.Sequential(
            nn.Identity(),
            InvertedResidual(InvertedResidualConfig(ouc, 3, ouc, ouc, True, "HS", 1, 1, 1), layer_norm, se_layer),
            InvertedResidual(InvertedResidualConfig(ouc, 3, ouc, ouc, True, "HS", 2, 1, 1), layer_norm, se_layer)
        )

    def forward(self, x):
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        x = self.mlp(x) 
        x = x.permute(0, 2, 1).reshape(b, -1, h, h)
        x = self.mb_block(x)
        x = x.flatten(2).permute(0, 2, 1)
        return x

class FeatureIRLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class TokenDownLayer(nn.Module):
    def __init__(self, shape) -> None:
        super().__init__()
        self.dwn = nn.Sequential(
            nn.AdaptiveAvgPool2d(shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        assert h * h == num_tokens
        x = x.permute(0, 2, 1).reshape(b, -1, h, h)
        x = self.dwn(x)
        x = x.flatten(2).transpose(1, 2)
        return x
    
class PosInjectLayer(nn.Module):
    # https://github.com/Meituan-AutoML/Twins/blob/main/gvt.py
    def __init__(self, in_dim: int, out_dim: int, stride: int = 1) -> None:
        super().__init__()
        self.peg = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride, 1, bias=True, groups=out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        assert h * h == num_tokens
        cnn_feat = x.transpose(1, 2).view(b, c, h, h)
        x = self.peg(cnn_feat) + cnn_feat
        x = x.flatten(2).transpose(1, 2)
        return x

class LDPNetProjector(nn.Module):
    
    def __init__(self, config=None):
        super().__init__()
        self.model = LDPBlock(config)

    def forward(self, x):
        return self.model(x)

class LDPNetV2Projector(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        inc, ouc = config.mm_hidden_size, config.hidden_size
        self.mlp = FeatureIRLayer(inc, ouc)
        self.dwn = TokenDownLayer((4, 4))
        self.peg = PosInjectLayer(ouc, ouc, stride=1)

    def forward(self, x):
        x = self.mlp(x)
        x = self.dwn(x)
        x = self.peg(x)
        return x

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


def build_vision_projector(config, **kwargs):
    """
        mm_hidden_size = 1408 for InternVideo2-Stage2_1B-224p-f4 (TODO: Update it if you use a different video encoder)
    """
    image_mm_projector = kwargs['image_mm_projector']
    if image_mm_projector:
        config.mm_hidden_size = 1024
        projector_type = getattr(config, 'basepruner_type', 'mlp2x_gelu')
    else:
        config.mm_hidden_size = 1408
        projector_type = getattr(config, 'basepruner_type', 'mlp2x_gelu')
    print(f"Building {projector_type}")

    if projector_type == 'linear':
        projector = nn.Linear(config.mm_hidden_size, config.hidden_size)
        config.mm_hidden_size = 1408
        config.image_mm_hidden_size = 1024
        return projector

    print("projector_type:", projector_type)
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        config.mm_hidden_size = 1408
        config.image_mm_hidden_size = 1024
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        projector = IdentityMap()
        config.mm_hidden_size = 1408
        config.image_mm_hidden_size = 1024
        return projector
    
    if projector_type.startswith('ldpv2'):
        return LDPNetV2Projector(config)
    if projector_type.startswith('ldp'):
        return LDPNetProjector(config)

    raise ValueError(f'Unknown projector type: {projector_type}')

# connect visual and cap_llm
def build_cap_vision_projector(config, **kwargs):
    """
        cap_hidden_state: 896 (TODO: Update it if you use a different light_llm)   
        mm_hidden_size = 1408 for InternVideo2-Stage2_1B-224p-f4 (TODO: Update it if you use a different video encoder)
    """
    # NOTE: rebuttal change here
    # config.cap_hidden_size = 1536
    image_mm_projector = kwargs['image_mm_projector']
    if image_mm_projector:
        projector_type = getattr(config, 'image_mm_projector_type', 'linear')
        config.mm_hidden_size = 1024
    else:
        config.mm_hidden_size = 1408
        projector_type = getattr(config, 'mm_projector_type', 'linear')
    print(f"Building {projector_type}")

    if projector_type == 'linear':
        projector = nn.Linear(config.mm_hidden_size, config.cap_hidden_size)
        config.mm_hidden_size = 1408
        config.image_mm_hidden_size = 1024
        return projector

    print("projector_type:", projector_type)
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.cap_hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.cap_hidden_size, config.cap_hidden_size))
        config.mm_hidden_size = 1408
        config.image_mm_hidden_size = 1024
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        projector = IdentityMap()
        config.mm_hidden_size = 1408
        config.image_mm_hidden_size = 1024
        return projector

    raise ValueError(f'Unknown projector type: {projector_type}')

# connect cap_llm and vqa_llm
def build_vqa_llm_projector(config, **kwargs):
    """
        cap_hidden_state: 896 (TODO: Update it if you use a different cappruner)   
    """
    # return nn.Linear(config.cap_hidden_size, config.hidden_size)
    projector_type = getattr(config, 'llm_projector_type', 'linear')
    if projector_type == 'linear':
        projector = nn.Linear(config.cap_hidden_size, config.hidden_size)
        return projector
    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.cap_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        projector = IdentityMap()
        return projector

    raise ValueError(f'Unknown projector type: {projector_type}')


def build_light_llm(config, **kwargs):
    cap_config = Qwen2Config.from_pretrained(config.cap_light_llm)
    cap_light_llm = Qwen2ForCausalLM(cap_config)
    cap_light_llm = Qwen2ForCausalLM.from_pretrained(config.cap_light_llm, torch_dtype=torch.float16)
    # cap_light_llm = QwenCapForCausalLM(cap_config)
    # cap_light_llm = QwenCapForCausalLM.from_pretrained(config.cap_light_llm, torch_dtype=torch.float16)
    return cap_light_llm