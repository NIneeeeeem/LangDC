from abc import ABC, abstractmethod
import torch
from .multimodal_encoder.builder import build_vision_tower
from langdc.constants import *
from .multimodal_projector.builder import build_vision_projector, build_light_llm, build_cap_vision_projector, build_vqa_llm_projector
from transformers import Qwen2Model, Qwen2Config
from einops import rearrange
import math, re
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import List, Optional, Tuple, Union

CHUNK_NUM = 4
CHUNK_SIZE = 4

def find_subsequence_index(main_tensor, subsequence):
    main_len = main_tensor.size(0)
    sub_len = subsequence.size(0)
    
    for i in range(main_len - sub_len + 1):
        candidate = main_tensor[i:i + sub_len]
        if torch.equal(candidate, subsequence):
            return i + sub_len + 1
    return -1

def apply_adaptive_avg_pooling(x, shape=(12, 12)):
    b, num_tokens, c = x.shape
    h = int(math.sqrt(num_tokens))
    assert h * h == num_tokens
    x = x.permute(0, 2, 1).reshape(b, -1, h, h)
    x = F.adaptive_avg_pool2d(x, shape)
    x = x.flatten(2).transpose(1, 2)
    return x

def load_projector_weights(projector, weights_path, keyword):
    if weights_path is not None:
        print(f"Initializing projector from {weights_path}")
        weights = torch.load(weights_path, map_location='cpu')
        filtered_weights = {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
        projector.load_state_dict(filtered_weights)

class MetaModel:
    def __init__(self, config):
        super(MetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True, image_vision_tower=False)
            self.image_vision_tower = build_vision_tower(config, delay_load=True, image_vision_tower=True)
            self.mm_projector = build_vision_projector(config, image_mm_projector=False)
            self.image_mm_projector = build_vision_projector(config, image_mm_projector=True)
            self.cap_mm_projector = build_cap_vision_projector(config, image_mm_projector=False)
            self.cap_image_mm_projector = build_cap_vision_projector(config, image_mm_projector=True)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_image_vision_tower(self):
        image_vision_tower = getattr(self, 'image_vision_tower', None)
        if type(image_vision_tower) is list:
            image_vision_tower = image_vision_tower[0]
        return image_vision_tower
    
    def _setup_config_attributes(self, model_args):
        self.config.mm_vision_tower = model_args.vision_tower
        self.config.image_mm_vision_tower = model_args.image_vision_tower
        self.config.mm_vision_select_layer = model_args.mm_vision_select_layer
        self.config.mm_vision_select_feature = model_args.mm_vision_select_feature
        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.image_mm_projector_type = getattr(model_args, 'image_mm_projector_type', 'linear')
        self.config.basepruner_type = getattr(model_args, 'basepruner_type', 'mlp2x_gelu')
        self.config.cap_hidden_size = getattr(model_args, 'cap_hidden_size', 896)

    def _build_vision_towers(self, model_args, has_pool_branch, has_cap_branch):
        if model_args.vision_tower is not None:
            vision_tower = build_vision_tower(model_args, image_vision_tower=False)
            self.config.mm_hidden_size = vision_tower.hidden_size
            if not hasattr(self, 'mm_projector') and has_pool_branch:
                self.mm_projector = build_vision_projector(self.config, image_mm_projector=False)
            if not hasattr(self, 'cap_mm_projector') and has_cap_branch:
                self.cap_mm_projector = build_cap_vision_projector(self.config, image_mm_projector=False)
        
        if model_args.image_vision_tower is not None:
            image_vision_tower = build_vision_tower(model_args, image_vision_tower=True)
            self.config.image_mm_hidden_size = image_vision_tower.hidden_size
            if not hasattr(self, 'image_mm_projector') and has_pool_branch:
                self.image_mm_projector = build_vision_projector(self.config, image_mm_projector=True)
            if not hasattr(self, 'cap_image_mm_projector') and has_cap_branch:
                self.cap_image_mm_projector = build_cap_vision_projector(self.config, image_mm_projector=True)
        
        return vision_tower if model_args.vision_tower else None, image_vision_tower if model_args.image_vision_tower else None

    def initialize_vision_modules(self, model_args, fsdp=None):
        """优化后的视觉模块初始化方法"""
        has_cap_branch = model_args.is_cap_branch
        has_pool_branch = model_args.is_pool_branch
        
        self._setup_config_attributes(model_args)
        vision_tower, image_vision_tower = self._build_vision_towers(model_args, has_pool_branch, has_cap_branch)

        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower] if vision_tower else None
            self.image_vision_tower = [image_vision_tower] if image_vision_tower else None
        else:
            self.vision_tower = vision_tower
            self.image_vision_tower = image_vision_tower
        load_projector_weights(self.mm_projector, model_args.pretrain_mm_mlp_adapter, '.mm_projector')
        load_projector_weights(self.image_mm_projector, model_args.pretrain_image_mm_mlp_adapter, 'image_mm_projector')
        load_projector_weights(self.cap_mm_projector, model_args.pretrain_cap_mm_mlp_adapter, '.cap_mm_projector')
        load_projector_weights(self.cap_image_mm_projector, model_args.pretrain_cap_image_mm_mlp_adapter, 'cap_image_mm_projector')
            
            
    def initialize_vision_modules_vqa(self, model_args, fsdp=None):
        has_cap_branch = model_args.is_cap_branch
        has_pool_branch = model_args.is_pool_branch  
        self._setup_config_attributes(model_args)
        vision_tower, image_vision_tower = self._build_vision_towers(model_args, has_pool_branch, has_cap_branch)
                
        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower] if vision_tower else None
            self.image_vision_tower = [image_vision_tower] if image_vision_tower else None
        else:
            self.vision_tower = vision_tower
            self.image_vision_tower = image_vision_tower
            
        if has_cap_branch:
            self.initialize_vision_modules_cap_branch(model_args, fsdp)
        if has_pool_branch:
            self.initialize_vision_modules_pool_branch(model_args, fsdp)
            
        
    def initialize_vision_modules_pool_branch(self, model_args, fsdp=None):
        self.config.use_mm_proj = True
        load_projector_weights(self.mm_projector, model_args.pretrain_mm_mlp_adapter, '.mm_projector')
        load_projector_weights(self.image_mm_projector, model_args.pretrain_image_mm_mlp_adapter, 'image_mm_projector')
        
    def initialize_vision_modules_cap_branch(self, model_args, fsdp=None):
        self.config.use_mm_proj = True
        load_projector_weights(self.cap_mm_projector, model_args.pretrain_cap_mm_mlp_adapter, '.cap_mm_projector')
        load_projector_weights(self.cap_image_mm_projector, model_args.pretrain_cap_image_mm_mlp_adapter, 'cap_image_mm_projector')
        
        self.cap_light_llm = build_light_llm(model_args)
        self.config.cap_hidden_size = getattr(model_args, 'cap_hidden_size', 896)
        self.config.cap_vocab_size = getattr(model_args, 'cap_vocab_size', 151936)
        self.config.llm_projector_type = getattr(model_args, 'llm_projector_type', 'linear')
        self.llm_projector = build_vqa_llm_projector(self.config)
        
        if model_args.pretrain_cap_vqa_adapter is not None:
            weights = torch.load(model_args.pretrain_cap_vqa_adapter)
            new_weight = {re.sub(r'^mlp\.', '', key): value for key, value in weights.items()}
            self.llm_projector.load_state_dict(new_weight)


class VideoGPTPlusMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_image_vision_tower(self):
        return self.get_model().get_image_vision_tower()

    def encode_images(self, images):
        image_encoder = self.get_model().get_image_vision_tower()
        video_encoder = self.get_model().get_vision_tower()
        if image_encoder is not None:
            image_features = image_encoder(images, select_feature="patch")
        elif video_encoder is not None:
            image_features = video_encoder(images.unsqueeze(1))  # Adds time dimension (B, T, C, H, W)
            image_features = image_features[:, 1:]

        return image_features

    def encode_videos_frames(self, frames_o, batch_size):
        frames = rearrange(frames_o, '(b t) c h w -> b t c h w', b=batch_size)
        num_chunks = frames.shape[1] // CHUNK_SIZE
        L = 256  # Number of features per frame from InternVideo2-Stage2_1B-224p-f4
        D = 1408  # Feature dimension of InternVideo2-Stage2_1B-224p-f4
        if num_chunks < 1:
            num_chunks = 1
            video_features = torch.zeros(batch_size, num_chunks, L, D, device=frames.device, dtype=frames.dtype)
            # frames = rearrange(frames, 'b t c h w -> (b t) c h w', b=batch_size)
            video_features = self.get_model().get_vision_tower()(frames)[:, 1:]
        else:
            batch_size, t, c, h, w = frames.shape
            video_features = torch.zeros(batch_size, num_chunks, CHUNK_SIZE * L, D, device=frames.device, dtype=frames.dtype)
            chunked_frames = frames.view(batch_size * num_chunks, CHUNK_SIZE, c, h, w)  # (batch_size * num_chunks, 4, c, h, w)
            chunk_features = self.get_model().get_vision_tower()(chunked_frames)  # (batch_size * num_chunks, 4*L, D)
            video_features = chunk_features[:, 1:]  # remove cls token 
        return video_features
    
    def encode_videos_context(self, context_images, batch_size):
        context_image_features = self.get_model().get_image_vision_tower()(context_images, select_feature="patch")
        return rearrange(context_image_features, '(b t) l d -> b t l d', b=batch_size)

    def positional_encoding(self, x, num_features=1024, max_len=64):
        p = torch.zeros((1, max_len, num_features))
        _x = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_features, 2, dtype=torch.float32) / num_features
        )

        p[:, :, 0::2] = torch.sin(_x)
        p[:, :, 1::2] = torch.cos(_x)
        x = x + p[:, :x.shape[1], :].to(x.device).to(x.dtype)
        return x
    # for cap pruner
    def project(self, video_features, context_features=None, input_type="image"):
        if input_type == "video":
            # NOTE: HALF()
            video_features = self.get_model().cap_mm_projector(video_features)
            video_features = rearrange(video_features, 'b (t l) d -> (b t) l d', t=CHUNK_NUM)  # t=4 - chunk size
            video_features = apply_adaptive_avg_pooling(video_features, shape=(8, 8))  # Feature pooling from 256 to 64
            video_features = rearrange(video_features, '(b t) l d -> b (t l) d', t=CHUNK_NUM)  # t=4 - chunk size

            context_image_features = self.get_model().cap_image_mm_projector(context_features)
            context_image_features = apply_adaptive_avg_pooling(context_image_features,
                                                                shape=(12, 12))  # Feature pooling from 576 to 144
            context_image_features = rearrange(context_image_features, '(b t) l d -> b (t l) d',
                                               b=video_features.shape[0])

            # merged with image feature in front
            merged_features = torch.cat((context_image_features, video_features), dim=1)
            return merged_features

        image_encoder = self.get_model().get_image_vision_tower()
        video_encoder = self.get_model().get_vision_tower()
        if image_encoder is not None and video_encoder is not None:
            video_features = self.get_model().cap_mm_projector(video_features)
            video_features = rearrange(video_features, 'b (t l) d -> (b t) l d', t=1)  # t=4 - chunk size
            video_features = apply_adaptive_avg_pooling(video_features, shape=(8, 8))  # Feature pooling from 256 to 64
            video_features = rearrange(video_features, '(b t) l d -> b (t l) d', t=1)  # t=4 - chunk size

            context_image_features = self.get_model().cap_image_mm_projector(context_features)
            context_image_features = apply_adaptive_avg_pooling(context_image_features,
                                                                shape=(12, 12))  # Feature pooling from 576 to 144
            context_image_features = rearrange(context_image_features, '(b t) l d -> b (t l) d',
                                               b=video_features.shape[0])

            context_features = torch.cat((context_image_features, video_features), dim=1)
        elif image_encoder is not None:
            context_features = self.get_model().cap_image_mm_projector(context_features)
        elif video_encoder is not None:
            context_features = self.get_model().cap_mm_projector(context_features)
        else:
            raise NotImplementedError("Either image_encoder or video_encoder should not be None.")

        return context_features

    # base pruner: Pooling
    # TODO: alter to ldpv2 or others
    def video_project_pool(self, video_features, context_features=None, input_type="image"):
        if input_type == "video":
            # NOTE: half()
            video_features = self.get_model().mm_projector(video_features)
            video_features = rearrange(video_features, 'b (t l) d -> (b t) l d', t=CHUNK_NUM)  # t=4 - chunk size
            video_features = apply_adaptive_avg_pooling(video_features, shape=(4, 4))  # Feature pooling from 256 to 64
            video_features = rearrange(video_features, '(b t) l d -> b (t l) d', t=CHUNK_NUM)  # t=4 - chunk size

            context_image_features = self.get_model().image_mm_projector(context_features)
            context_image_features = apply_adaptive_avg_pooling(context_image_features,
                                                                shape=(6, 6))  # Feature pooling from 576 to 144
            context_image_features = rearrange(context_image_features, '(b t) l d -> b (t l) d',
                                               b=video_features.shape[0])

            merged_features = []
            for i in range(context_image_features.shape[0]):
                merged_features.append(context_image_features[i])

            for i in range(video_features.shape[0]):
                merged_features.append(video_features[i])

            merged_features = torch.cat(merged_features, dim=0).unsqueeze(0)

            return merged_features
        
        image_encoder = self.get_model().get_image_vision_tower()
        video_encoder = self.get_model().get_vision_tower()

        if image_encoder is not None:
            context_features = self.get_model().image_mm_projector(video_features)
        elif video_encoder is not None:
            context_features = self.get_model().mm_projector(video_features)
        else:
            raise NotImplementedError("Either image_encoder or video_encoder should not be None.")

        return context_features
    
    def prepare_inputs_labels_for_multimodal(self, input_ids, attention_mask, past_key_values, labels, images,
                                             context_images, pool_branch=False):
        vision_tower = self.get_vision_tower()
        image_vision_tower = self.get_image_vision_tower()
        if (vision_tower is None and image_vision_tower is None) or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[
                1] == 1:
                attention_mask = torch.ones(
                    (attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
            return input_ids, attention_mask, past_key_values, None, labels

        if images is not None and context_images is not None:
            video_features = self.encode_videos_frames(images, batch_size=input_ids.shape[0])
            context_features = self.encode_videos_context(context_images, batch_size=input_ids.shape[0])
        elif images is not None: # pretraining branch
            image_features = self.encode_images(images)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # Multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = cur_input_embeds + (
                        0. * self.get_model().mm_projector(vision_tower.dummy_feature)).sum()
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]

            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape

            if len(image_token_indices) > 1:  # This is a video
                temp = []
                cur, pre = image_token_indices[0], image_token_indices[0]
                for i in image_token_indices:
                    cur = i
                    if cur - pre == 1:
                        temp[-1] = temp[-1] + [cur]
                    else:
                        temp.append([cur])
                    pre = cur

                for i in temp:
                    image_token_start = image_token_indices[0]
                    image_token_end = image_token_indices[-1]
                    cur_image_features = []

                    for _ in range(len(i) // CHUNK_SIZE):
                        cur_image_features.append(video_features[cur_image_idx])
                        cur_image_idx += 1

                    if len(i) > 2:
                        cur_image_features = torch.stack(cur_image_features, dim=0)
                        if pool_branch:
                            cur_image_features = self.video_project_pool(cur_image_features, context_features[batch_idx],
                                                          input_type="video")
                        else:
                            cur_image_features = self.project(cur_image_features, context_features[batch_idx],
                                                          input_type="video")
                        t, l, n = cur_image_features.size()
                        cur_image_features = cur_image_features.contiguous().view(t * l, n)
                    else:
                        # This is video but only 1 frame is sampled
                        # This will not happen as video encoder needs at least 4 frames
                        cur_image_features = torch.stack(cur_image_features, dim=0)
                        if pool_branch:
                            cur_image_features = self.video_project_pool(cur_image_features, context_features[batch_idx],
                                                          input_type="image")
                        else: 
                            cur_image_features = self.project(cur_image_features, context_features[batch_idx],
                                                          input_type="image")
                        t, l, n = cur_image_features.size()
                        cur_image_features = cur_image_features.contiguous().view(t * l, n)

                    if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(
                            self.config, 'mm_use_im_start_end', False
                    ):
                        cur_new_input_embeds.append(
                            self.get_model().embed_tokens(cur_input_ids[:image_token_start - 1]).detach()
                        )
                        cur_new_input_embeds.append(
                            self.get_model().embed_tokens(cur_input_ids[image_token_start - 1:image_token_start])
                        )
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_input_embeds.append(
                            self.get_model().embed_tokens(cur_input_ids[image_token_end + 1:image_token_end + 2])
                        )
                        if labels is not None:
                            cur_new_labels.append(cur_labels[:image_token_start])
                            cur_new_labels.append(
                                torch.full(
                                    (cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device,
                                    dtype=labels.dtype
                                )
                            )
                            cur_new_labels.append(cur_labels[image_token_end:image_token_end + 1])
                            cur_labels = cur_labels[image_token_end + 2:]
                    else:
                        cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                        cur_new_input_embeds.append(cur_image_features)
                        if labels is not None:
                            cur_new_labels.append(cur_labels[:image_token_start])
                            cur_new_labels.append(
                                torch.full(
                                    (cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device,
                                    dtype=labels.dtype
                                )
                            )
                            cur_labels = cur_labels[image_token_end + 1:]

                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(
                        self.config, 'mm_use_im_start_end', False
                ):
                    cur_input_ids = cur_input_ids[image_token_end + 2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_end + 1:]

            elif image_token_indices.numel() > 0:  # This is an image
                cur_image_features = []
                image_token_start = image_token_indices[0]
                image_token_end = image_token_indices[-1]

                for _ in image_token_indices:
                    cur_image_features.append(image_features[cur_image_idx])
                    cur_image_idx += 1

                cur_image_features = torch.stack(cur_image_features, dim=0)
                # cur_image_features = self.project(video_features=None, context_features=cur_image_features, input_type="image")
                if hasattr(self.get_model(), 'cap_image_mm_projector') or hasattr(self.get_model(), 'cap_mm_projector'):
                    cur_image_features = self.project(video_features=video_features, context_features=cur_image_features, input_type="image")
                else:
                    cur_image_features = self.video_project_pool(video_features=cur_image_features, input_type="image")
                t, l, n = cur_image_features.size()
                cur_image_features = cur_image_features.contiguous().view(t * l, n)

                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(
                        self.config, 'mm_use_im_start_end', False
                ):
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[:image_token_start - 1]).detach()
                    )
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[image_token_start - 1:image_token_start])
                    )
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[image_token_end + 1:image_token_end + 2])
                    )
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype
                            )
                        )
                        cur_new_labels.append(cur_labels[image_token_end:image_token_end + 1])
                        cur_labels = cur_labels[image_token_end + 2:]
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype
                            )
                        )
                        cur_labels = cur_labels[image_token_end + 1:]

                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(
                        self.config, 'mm_use_im_start_end', False
                ):
                    cur_input_ids = cur_input_ids[image_token_end + 2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_end + 1:]

            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(
                        self.config, 'mm_use_im_start_end', False
                ):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat(
                    (cur_new_embed, torch.zeros(
                        (max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                        device=cur_new_embed.device
                    )), dim=0
                )
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat(
                        (cur_new_label, torch.full(
                            (max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype,
                            device=cur_new_label.device
                        )), dim=0
                    )
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(
                        attention_mask, _new_labels, new_labels
                ):
                    new_attn_mask_pad_left = torch.full(
                        (cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype,
                        device=attention_mask.device
                    )
                    new_attn_mask_pad_right = torch.full(
                        (cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype,
                        device=attention_mask.device
                    )
                    cur_new_attention_mask = torch.cat(
                        (new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0
                    )
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True,
                    dtype=attention_mask.dtype, device=attention_mask.device
                )
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    
    def prepare_inputs_labels_for_cap(self, input_ids, attention_mask, past_key_values, labels, images, context_images
                                             ):
        vision_tower = self.get_vision_tower()
        image_vision_tower = self.get_image_vision_tower()
        if (vision_tower is None and image_vision_tower is None) or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[
                1] == 1:
                attention_mask = torch.ones(
                    (attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype,
                    device=attention_mask.device
                )
            return input_ids, attention_mask, past_key_values, None, labels

        if images is not None and context_images is not None:
            video_features = self.encode_videos_frames(images, batch_size=input_ids.shape[0])
            context_features = self.encode_videos_context(context_images, batch_size=input_ids.shape[0])
        elif images is not None:
            image_features = self.encode_images(images)

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # Multimodal LLM, but the current sample is not multimodal
                cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = cur_input_embeds + (
                        0. * self.get_model().mm_projector(vision_tower.dummy_feature)).sum()
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]

            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape

            if len(image_token_indices) > 1:  # This is a video / image
                temp = []
                cur, pre = image_token_indices[0], image_token_indices[0]
                for i in image_token_indices:
                    cur = i
                    if cur - pre == 1:
                        temp[-1] = temp[-1] + [cur]
                    else:
                        temp.append([cur])
                    pre = cur

                for i in temp: # frame nums
                    image_token_start = image_token_indices[0]
                    image_token_end = image_token_indices[-1]
                    cur_image_features = []

                    for _ in range(len(i) // CHUNK_SIZE):
                        cur_image_features.append(video_features[cur_image_idx])
                        cur_image_idx += 1

                    if len(i) > 2:
                        cur_image_features = torch.stack(cur_image_features, dim=0)
                        cur_image_features = self.project(cur_image_features, context_features[batch_idx],
                                                          input_type="video")
                        t, l, n = cur_image_features.size()
                        cur_image_features = cur_image_features.contiguous().view(t * l, n)
                    else:
                        # This is video but only 1 frame is sampled
                        # This will not happen as video encoder needs at least 4 frames
                        cur_image_features = torch.stack(cur_image_features, dim=0)
                        cur_image_features = self.project(cur_image_features, context_features[batch_idx],
                                                          input_type="image")
                        t, l, n = cur_image_features.size()
                        cur_image_features = cur_image_features.contiguous().view(t * l, n)

                    if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(
                            self.config, 'mm_use_im_start_end', False
                    ):
                        cur_new_input_embeds.append(
                            self.get_model().cap_light_llm.model.embed_tokens(cur_input_ids[:image_token_start - 1]).detach()
                        )
                        cur_new_input_embeds.append(
                            self.get_model().cap_light_llm.model.embed_tokens(cur_input_ids[image_token_start - 1:image_token_start])
                        )
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_input_embeds.append(
                            self.get_model().cap_light_llm.model.embed_tokens(cur_input_ids[image_token_end + 1:image_token_end + 2])
                        )
                        if labels is not None:
                            cur_new_labels.append(cur_labels[:image_token_start])
                            cur_new_labels.append(
                                torch.full(
                                    (cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device,
                                    dtype=labels.dtype
                                )
                            )
                            cur_new_labels.append(cur_labels[image_token_end:image_token_end + 1])
                            cur_labels = cur_labels[image_token_end + 2:]
                    else:
                        # BUG: 区分vqa branch和 vcap branch，否则问题：vcap训练时 
                        if hasattr(self.get_model(), 'cap_light_llm'): 
                            if self.get_model().cap_light_llm is not None: # this is for vqa branch
                                cur_new_input_embeds.append(self.get_model().cap_light_llm.model.embed_tokens(cur_input_ids[:image_token_start]))
                                cur_new_input_embeds.append(cur_image_features)
                            else: # vcap inference branch
                                cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                                cur_new_input_embeds.append(cur_image_features)
                        else: # this is for vcap training branch
                            cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                            cur_new_input_embeds.append(cur_image_features)

                        if labels is not None:
                                cur_new_labels.append(cur_labels[:image_token_start])
                                cur_new_labels.append(
                                    torch.full(
                                        (cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device,
                                        dtype=labels.dtype
                                    )
                                )
                                cur_labels = cur_labels[image_token_end + 1:]

                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(
                        self.config, 'mm_use_im_start_end', False
                ):
                    cur_input_ids = cur_input_ids[image_token_end + 2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_end + 1:]
            elif image_token_indices.numel() > 0:  # This is an image
                cur_image_features = []
                cur_video_features = []
                image_token_start = image_token_indices[0]
                image_token_end = image_token_indices[-1]

                cur_image_features=context_features[cur_image_idx]
                cur_video_features=video_features[cur_image_idx].unsqueeze(dim=0)
                cur_image_idx += 1

                # cur_image_features = torch.stack(cur_image_features, dim=0).squeeze(dim=0)
                # cur_image_features = self.project(video_features=None, context_features=cur_image_features, input_type="image")
                if hasattr(self.get_model(), 'cap_image_mm_projector') or hasattr(self.get_model(), 'cap_mm_projector'):
                    cur_image_features = self.project(video_features=cur_video_features, context_features=cur_image_features, input_type="image")
                else:
                    cur_image_features = self.video_project_pool(video_features=cur_video_features, context_features=cur_image_features, input_type="image")
                # cur_image_features = apply_adaptive_avg_pooling(cur_image_features, shape=(8, 8))
                t, l, n = cur_image_features.size()
                cur_image_features = cur_image_features.contiguous().view(t * l, n)

                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(
                        self.config, 'mm_use_im_start_end', False
                ):
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[:image_token_start - 1]).detach()
                    )
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[image_token_start - 1:image_token_start])
                    )
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(
                        self.get_model().embed_tokens(cur_input_ids[image_token_end + 1:image_token_end + 2])
                    )
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype
                            )
                        )
                        cur_new_labels.append(cur_labels[image_token_end:image_token_end + 1])
                        cur_labels = cur_labels[image_token_end + 2:]
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(
                            torch.full(
                                (cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype
                            )
                        )
                        cur_labels = cur_labels[image_token_end + 1:]

                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(
                        self.config, 'mm_use_im_start_end', False
                ):
                    cur_input_ids = cur_input_ids[image_token_end + 2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_end + 1:]

            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(
                        self.config, 'mm_use_im_start_end', False
                ):
                    cur_new_input_embeds.append(self.get_model().cap_light_llm.model.embed_tokens(cur_input_ids).detach())
                else:
                    # BUG: DEBUG HERE
                    # for vqa branch
                    if hasattr(self.get_model(), 'cap_light_llm'):
                        if self.get_model().cap_light_llm is not None:
                            cur_new_input_embeds.append(self.get_model().cap_light_llm.model.embed_tokens(cur_input_ids))
                        else:
                            cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                    else:
                        cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat(
                    (cur_new_embed, torch.zeros(
                        (max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype,
                        device=cur_new_embed.device
                    )), dim=0
                )
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat(
                        (cur_new_label, torch.full(
                            (max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype,
                            device=cur_new_label.device
                        )), dim=0
                    )
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(
                        attention_mask, _new_labels, new_labels
                ):
                    new_attn_mask_pad_left = torch.full(
                        (cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype,
                        device=attention_mask.device
                    )
                    new_attn_mask_pad_right = torch.full(
                        (cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype,
                        device=attention_mask.device
                    )
                    cur_new_attention_mask = torch.cat(
                        (new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0
                    )
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True,
                    dtype=attention_mask.dtype, device=attention_mask.device
                )
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels
    
    
    def prepare_inputs_labels_for_cap_vqa(self, input_ids, attention_mask, past_key_values, labels, images, context_images
                                             ):
        video_features = self.encode_videos_frames(images, batch_size=input_ids.shape[0])
        context_features = self.encode_videos_context(context_images, batch_size=input_ids.shape[0])

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        image_token_indices = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0]
        image_token_start = image_token_indices[0]
        image_token_end = image_token_indices[-1]
        cur_image_features = self.project(video_features, context_features.reshape((-1,context_features.size(-2),context_features.size(-1))),
                                                          input_type="video")
        prompt_embeds = self.get_model().cap_light_llm.model.embed_tokens(input_ids[:,:image_token_start])
        question_embeds = self.get_model().cap_light_llm.model.embed_tokens(input_ids[:,image_token_end+1:])
        t, l, n = cur_image_features.size()
        new_input_embeds = torch.cat((prompt_embeds, cur_image_features, question_embeds), dim=1)
        if labels is not None:
            new_labels = torch.cat((labels[:,:image_token_start], 
                torch.full((t,l,), IGNORE_INDEX, device=labels.device,dtype=labels.dtype), labels[:,image_token_end + 1:]), dim=1)
        if attention_mask is not None:
            new_attn_mask_pad_left = torch.full(
                    (attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True,
                    dtype=attention_mask.dtype, device=attention_mask.device
                )
            attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
        return None, attention_mask, past_key_values, new_input_embeds, new_labels,video_features, context_features

    def prepare_inputs_labels_for_vqa_capbranch(self, input_ids, attention_mask, labels, input_ids_caps, attention_mask_cap, past_key_values, labels_caps, images,
                                             context_images, use_cache, hidden_state_layer=-1):
        """
            Input:  # list: cap_num * [batch_num * token]
            input_ids,attention_mask, labels: for vqa
            input_ids_caps,attention_mask_cap: for caption
            Output: captions [connected], attention mask
        """
        if past_key_values is not None and input_ids.shape[1] == 1:
            attention_mask = torch.ones(
                (attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            return input_ids, attention_mask, past_key_values, None, None, 0 # loss
        
        batch_size = input_ids.size(0)
        cap_loss = 0
        # same prompt for caption now. TODO: get cap_start_index with attention map
        if labels_caps is not None: # this is for train
            cap_start_idx = torch.where(labels_caps[0] != IGNORE_INDEX)[0][0] # tensor(66, device='cuda:0')
            cap_start = torch.full((labels_caps.size(0),), cap_start_idx).to(input_ids_caps[0].device) # tensor([66, 66], device='cuda:0')           
            caption_input_ids, caption_attention_mask, caption_past_key_values, caption_inputs_embeds, caption_labels,_,_ = self.prepare_inputs_labels_for_cap_vqa(
                input_ids_caps, attention_mask_cap, past_key_values, labels_caps, images, context_images)
            with torch.no_grad():
                outputs = self.get_model().cap_light_llm(
                    input_ids=caption_input_ids,
                    attention_mask=caption_attention_mask,
                    past_key_values=caption_past_key_values,
                    inputs_embeds=caption_inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                    # hidden_state_layer=hidden_state_layer
                )
            logits = outputs[0] # [bs*CHUNK_NUM, token_num, token_dim] torch.Size([8, 958, 151936])
            hidden_states = outputs.hidden_states # [model_layer, bs*CHUNK_SIZE, token_num, dim]
            cap_start = cap_start + (logits.size(-2)-input_ids_caps.size(-1)) # add visual tokens
            cap_end = logits.size(1) - torch.argmax(torch.flip(caption_attention_mask.to(torch.int), dims=[1]), dim=1)
                # get cap new_input_embeds and attention_mask
            # NOTE: Another way to get cap_attention is setting output_attentions=True, however this may cause OOM
            
            new_input_embeds = hidden_states[hidden_state_layer][:,cap_start[0]:max(cap_end+1),:].reshape(batch_size,-1,hidden_states[hidden_state_layer].size(-1)) # 每个 cap 没有加 padding
            # new_input_embeds = hidden_states[0][:,cap_start[0]:max(cap_end+1),:].reshape(batch_size,-1,hidden_states[0].size(-1))
            
            attention_num = cap_end - cap_start # BUG HERE -- fixed 1009
            max_len = max(attention_num)
            attention_mask_caption = torch.arange(max_len, device=attention_num.device).expand(len(attention_num), max_len) < torch.tensor(attention_num,device=attention_num.device).unsqueeze(1)
            attention_mask_caption = attention_mask_caption.reshape(batch_size,-1).cpu()
                
        else:
            caption_input_ids, caption_attention_mask, caption_past_key_values, caption_inputs_embeds, caption_labels,_,_ = self.prepare_inputs_labels_for_cap_vqa(
                input_ids_caps, attention_mask_cap, past_key_values, labels_caps, images, context_images)
            with torch.no_grad():
                self.get_model().cap_light_llm.gradient_checkpointing_disable()
                outputs = self.get_model().cap_light_llm.generate(
                    input_ids=caption_input_ids, # None
                    # attention_mask=caption_attention_mask, # None
                    # past_key_values=caption_past_key_values, # None
                    inputs_embeds=caption_inputs_embeds, # torch.Size([1, 879, 896])
                    use_cache=True, # True
                    output_attentions=False, # False
                    output_hidden_states=True, # False
                    return_dict_in_generate = True,
                    max_new_tokens = 128,
                    # repetition_penalty = 1.0,
                    # hidden_state_layer = hidden_state_layer
                )
            logits = outputs[0]
            
            new_input_embeds = torch.stack([hidden_state[hidden_state_layer][:, -1, :] for hidden_state in outputs.hidden_states], dim=1) #[8, 65, 896]
            # new_input_embeds = torch.stack([hidden_state[0][:, -1, :] for hidden_state in outputs.hidden_states], dim=1)
            new_input_embeds = new_input_embeds.reshape(batch_size,-1,new_input_embeds.size(-1))
            logits_ref = logits.reshape(batch_size,-1)
            attention_mask_caption = torch.ones_like(logits_ref, dtype=torch.bool)
            attention_mask_caption[logits_ref == 151643] = False # TODO: here 151643 is the padding token idx for cappruner

        new_input_embed = self.get_model().llm_projector(new_input_embeds.clone())
        # new_input_embed = self.get_model().llm_projector(torch.cat(new_input_embeds, dim=1)) # debug here: nan
        # vqa_input_embeds = self.get_model().embed_tokens(input_ids)
        # NOTE: Here, our system prompt should always be same!
        image_token_indices = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0]
        system_input_embeds = self.get_model().embed_tokens(input_ids[:,:image_token_indices[0]])
        vqa_input_embeds = self.get_model().embed_tokens(input_ids[:,image_token_indices[-1]+1:])
        # TODO: here cap_token + vqa prompt token -> instruction + cap_token + question
        # new_input_embeds = torch.cat([vqa_input_embeds, new_input_embed], dim=1)
        new_input_embeds = torch.cat([system_input_embeds, new_input_embed, vqa_input_embeds], dim=1)
        # new_attention_mask = torch.cat([attention_mask, torch.cat(new_attention_mask, dim=1).to(attention_mask.device)], dim=1)
        new_attention_mask = torch.cat([attention_mask[:,:image_token_indices[0]], attention_mask_caption.to(attention_mask.device), attention_mask[:,image_token_indices[-1]+1:]], dim=1)
        # [bs, token_num, dim]
        # get new labels (length + captions)
        new_labels = None
        if labels is not None:
            padding_tensor = torch.full((batch_size, new_attention_mask.size(1)-labels.size(1)), IGNORE_INDEX).to(labels.device)
            new_labels = torch.cat((padding_tensor, labels), dim=1) 
        return None, new_attention_mask, past_key_values, new_input_embeds, new_labels, cap_loss
    
    def prepare_inputs_labels_for_vqa_poolbranch(self, input_ids, attention_mask, labels, input_ids_caps, attention_mask_cap, past_key_values, labels_caps, image_list,
                                             context_image_list, use_cache):
        """
            Input:  # list: cap_num * [batch_num * token]
            input_ids,attention_mask, labels: for vqa
            input_ids_caps,attention_mask_cap: for caption
            Output: captions [connected], attention mask
        """
        new_input_ids, new_attention_mask, past_key_values, new_input_embeds, new_labels= self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, image_list,
                                             context_image_list, pool_branch=True)
        return new_input_ids, new_attention_mask, past_key_values, new_input_embeds, new_labels, 0 
    
    def prepare_inputs_labels_for_vqa_dualbranch(self, input_ids, attention_mask, labels, input_ids_caps, attention_mask_cap, past_key_values, labels_caps, images,
                                             context_images, use_cache, hidden_state_layer=-1):
        """
            Input:  # list: cap_num * [batch_num * token]
            input_ids,attention_mask, labels: for vqa
            input_ids_caps,attention_mask_cap: for caption
            Output: captions [connected], attention mask
        """
        if past_key_values is not None and input_ids.shape[1] == 1:
            attention_mask = torch.ones(
                (attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype,
                device=attention_mask.device
            )
            return input_ids, attention_mask, past_key_values, None, None, 0 # loss
        
        # NOTE: Here, our system prompt should always be same!
        image_token_indices = torch.where(input_ids[0] == IMAGE_TOKEN_INDEX)[0]
        
        batch_size = input_ids.size(0)
    
        new_input_embeds = []
        new_attention_mask = []
        cap_loss = 0
        if labels_caps is not None: # train branch
            # TODO: same prompt for caption generation, change here later
            cap_start_idx = torch.where(labels_caps[0] != IGNORE_INDEX)[0][0] # tensor(66, device='cuda:0')
            cap_start = torch.full((labels_caps.size(0),), cap_start_idx).to(input_ids_caps[0].device) # tensor([66, 66], device='cuda:0') 
            # logits = self.get_model().light_lm_head(hidden_states) # [8, 876, 151936]
            # for i in range(len(input_ids_caps)): # this is for train
            caption_input_ids, caption_attention_mask, caption_past_key_values, caption_inputs_embeds, caption_labels,video_features, context_features_chunk = self.prepare_inputs_labels_for_cap_vqa(
                input_ids_caps, attention_mask_cap, past_key_values, labels_caps, images, context_images)
            with torch.no_grad():    
                outputs = self.get_model().cap_light_llm(
                    input_ids=caption_input_ids,
                    attention_mask=caption_attention_mask,
                    past_key_values=caption_past_key_values,
                    inputs_embeds=caption_inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=False,
                    output_hidden_states=True,
                    return_dict=True,
                    # hidden_state_layer = hidden_state_layer
                )
            logits = outputs[0] # [8, 876, 896] [bs, token_num, token_dim]
            hidden_states = outputs.hidden_states
            cap_start = cap_start + (logits.size(-2)-input_ids_caps.size(-1)) # add visual tokens
            cap_end = logits.size(1) - torch.argmax(torch.flip(caption_attention_mask.to(torch.int), dims=[1]), dim=1)
            new_input_embeds = hidden_states[0][:,cap_start[0]:max(cap_end+1),:].reshape(batch_size,-1,hidden_states[0].size(-1))
            # new_input_embeds.append(hidden_states[hidden_state_layer][:,cap_start[0]:max(cap_end+1),:]) # 每个 cap 没有加 padding
            attention_num = cap_end - cap_start 
            max_len = max(attention_num)
            attention_mask_caption = torch.arange(max_len, device=attention_num.device).expand(len(attention_num), max_len) < torch.tensor(attention_num, device=attention_num.device).unsqueeze(1)
            attention_mask_caption = attention_mask_caption.reshape(batch_size,-1).cpu()
        else: # this is for inference
            # TODO: cap inference batch > 1
            caption_input_ids, caption_attention_mask, caption_past_key_values, caption_inputs_embeds, caption_labels,video_features, context_features_chunk = self.prepare_inputs_labels_for_cap_vqa(
                    input_ids_caps, attention_mask_cap, past_key_values, labels_caps, images, context_images)
            outputs = self.get_model().cap_light_llm.generate(
                    input_ids=caption_input_ids, # None
                    # attention_mask=caption_attention_mask, # None
                    # past_key_values=caption_past_key_values, # None
                    inputs_embeds=caption_inputs_embeds, # torch.Size([1, 879, 896])
                    use_cache=use_cache, # True
                    output_attentions=False, # False
                    output_hidden_states=True, # False
                    return_dict_in_generate = True,
                    max_new_tokens = 256,
                    # hidden_state_layer = hidden_state_layer
                )
            logits = outputs[0]
            
            # new_input_embeds = torch.stack([hidden_state[hidden_state_layer][:, -1, :] for hidden_state in outputs.hidden_states], dim=1)
            new_input_embeds = torch.stack([hidden_state[0][:, -1, :] for hidden_state in outputs.hidden_states], dim=1)
            caps_out = []
            for id,cap_logits in enumerate(logits):
                # TODO: here 151643 is the padding token idx for qwen2
                indices = torch.where(cap_logits == 151643)[0]
                if indices.numel() > 0:
                    cap_end = indices[0]
                else:
                    cap_end = -1
                caps_out.append(new_input_embeds[id,:cap_end,:])
                # new_input_embeds.append(torch.stack(cap_hidden_states, dim=0).unsqueeze(dim=0))
            new_input_embeds = torch.concat(caps_out).unsqueeze(dim=0)
            print(new_input_embeds.shape[1])
            attention_mask_caption = torch.ones((1, len(new_input_embeds[0])), dtype=torch.bool)
        
        video_features = self.encode_videos_frames(images, batch_size=input_ids.shape[0])
        context_features = context_features_chunk.view(input_ids.shape[0],-1,context_features_chunk.size(-2),context_features_chunk.size(-1))
        visual_pool_feature = []
        for batch_idx, cur_input_ids in enumerate(input_ids):
            start_idx, end_idx = batch_idx * CHUNK_SIZE, (batch_idx + 1) * CHUNK_SIZE
            # cur_image_features = [video_features[i] for i in range(batch_idx*CHUNK_SIZE,(batch_idx+1)*CHUNK_SIZE)]
            cur_image_features = video_features[start_idx:end_idx]
            cur_image_features = self.video_project_pool(cur_image_features, context_features[batch_idx],
                                                          input_type="video")
            t, l, n = cur_image_features.size()
            cur_image_features = cur_image_features.contiguous().view(t * l, n)
            visual_pool_feature.append(cur_image_features)
        visual_pool_feature = torch.stack(visual_pool_feature, dim=0)
        visual_pool_attention = torch.ones((visual_pool_feature.size(0), visual_pool_feature.size(1)), dtype=torch.bool).to(attention_mask.device)

        new_input_embed = self.get_model().llm_projector(new_input_embeds)
        system_input_embeds = self.get_model().embed_tokens(input_ids[:,:image_token_indices[0]])
        vqa_input_embeds = self.get_model().embed_tokens(input_ids[:,image_token_indices[-1]+1:])
        # TODO: here cap_token + vqa prompt token -> instruction + cap_token + question
        # [bs, token_num, dim]
        new_input_embeds = torch.cat([system_input_embeds, visual_pool_feature,new_input_embed, vqa_input_embeds], dim=1)
        new_attention_mask = torch.cat([attention_mask[:,:image_token_indices[0]], visual_pool_attention, attention_mask_caption.to(attention_mask.device),attention_mask[:,image_token_indices[-1]+1:]], dim=1)
        # get new labels (length + captions)
        new_labels = None
        if labels_caps is not None:
            padding_tensor = torch.full((batch_size, new_attention_mask.size(1)-labels.size(1)), IGNORE_INDEX).to(labels.device)
            new_labels = torch.cat((padding_tensor, labels), dim=1) 
        return None, new_attention_mask, past_key_values, new_input_embeds, new_labels, cap_loss
    
    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_VID_START_TOKEN, DEFAULT_VID_END_TOKEN],
                special_tokens=True
            )
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter or model_args.tune_cap_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                print(f"Initializing projector from {model_args.pretrain_mm_mlp_adapter}")
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
