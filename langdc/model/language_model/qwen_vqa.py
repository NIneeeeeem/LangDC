from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Model, Qwen2Config, Qwen2ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from langdc.model.arch import MetaModel, VideoGPTPlusMetaForCausalLM
from langdc.model.language_model.qwen_cap import VideoQwenCapModel, VideoQwenCapForCausalLM, VideoQwenCapConfig
from langdc.constants import *


class VideoQwenVQAConfig(Qwen2Config):
    model_type = "VideoQwenVQA"


class VideoQwenVQAModel(MetaModel, Qwen2Model):
    config_class = VideoQwenVQAConfig

    def __init__(self, config: Qwen2Config):
        super(VideoQwenVQAModel, self).__init__(config)


class VideoQwenVQAForCausalLM(Qwen2ForCausalLM, VideoGPTPlusMetaForCausalLM):
    config_class = VideoQwenVQAConfig

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = VideoQwenVQAModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)    
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model
    
    def forward(self,
            input_ids: torch.LongTensor = None, # torch.Size([bs, token_length])
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None, # torch.Size([96, 3, 224, 224])
            context_images: Optional[torch.FloatTensor] = None, # torch.Size([96, 3, 336, 336])
            return_dict: Optional[bool] = None,
            input_ids_caps: Optional[torch.LongTensor] = None,
            labels_caps: Optional[torch.LongTensor] = None,
            attention_mask_cap: Optional[torch.LongTensor] = None,
            hidden_state_layer: Optional[int] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        alpha = getattr(self.config, 'alpha', 0)
        # cap_start_inputs_ids = getattr(self.config, 'cap_start_inputs_ids', torch.tensor([3846, 2821, 25, 220]))
        has_pool_branch = self.config.is_pool_branch
        has_cap_branch = self.config.is_cap_branch
        
        # hidden_state_layer 在训练和推理时都被使用
        hidden_state_layer = hidden_state_layer if hidden_state_layer is not None else self.config.hidden_state_layer
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # if past_key_values is not None:
            
        # else:
        # import pdb; pdb.set_trace()
        if has_pool_branch and has_cap_branch:
            caption_input_ids, caption_attention_mask, caption_past_key_values, caption_inputs_embeds,new_labels, cap_loss = self.prepare_inputs_labels_for_vqa_dualbranch(
                input_ids,attention_mask, labels, input_ids_caps, attention_mask_cap, past_key_values, labels_caps, images, context_images,use_cache,hidden_state_layer)
        elif has_cap_branch:
            caption_input_ids, caption_attention_mask, caption_past_key_values, caption_inputs_embeds,new_labels, cap_loss = self.prepare_inputs_labels_for_vqa_capbranch(
                input_ids,attention_mask, labels, input_ids_caps, attention_mask_cap, past_key_values, labels_caps, images, context_images,use_cache,hidden_state_layer)
        elif has_pool_branch:
            caption_input_ids, caption_attention_mask, caption_past_key_values, caption_inputs_embeds,new_labels, cap_loss = self.prepare_inputs_labels_for_vqa_poolbranch(
                input_ids,attention_mask, labels, input_ids_caps, attention_mask_cap, past_key_values, labels_caps, images, context_images,use_cache)
        outputs = self.model(
            input_ids=caption_input_ids,
            attention_mask=caption_attention_mask,
            past_key_values=caption_past_key_values,
            inputs_embeds=caption_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = 0
        if new_labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = new_labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable langdc/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            # return (loss+alpha*cap_loss,) + output if loss is not None else output
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithPast(
            # loss=loss+alpha*cap_loss,
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=hidden_states, # debug here
            attentions=outputs.attentions,
        )
    
    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "context_images": kwargs.get("context_images", None),
                "input_ids_caps": kwargs.get("input_ids_caps", None),
                "hidden_state_layer": kwargs.get("hidden_state_layer", -1),
            }
        )
        return model_inputs


AutoConfig.register("VideoQwenVQA", VideoQwenVQAConfig)
# AutoModelForCausalLM.register(VideoQwenVQAConfig, VideoQwenVQAForCausalLM)
AutoLigerKernelForCausalLM.register(VideoQwenVQAConfig, VideoQwenVQAForCausalLM)
