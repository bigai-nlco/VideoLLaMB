#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLMRMT(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        # super(LlavaLlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # # config memory tokens
        # num_mem_tokens = 0
        # self.create_memory(num_mem_tokens)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model
    
    # def create_memory(self, num_mem_tokens):
    #     self.num_mem_tokens = num_mem_tokens
    #     embeddings = self.model.get_input_embeddings()
    #     memory_dim = getattr(self.model.config, 'hidden_size', self.model.config.hidden_size)
    #     memory_weights = torch.randn((num_mem_tokens, memory_dim)) * embeddings.weight.data.std()
    #     self.register_parameter('memory', torch.nn.Parameter(memory_weights, requires_grad=True))

    #     self.read_memory_position = range(num_mem_tokens)
    #     self.write_memory_position = range(-num_mem_tokens, 0)
    
    # def set_memory(self, input_shape):
    #     memory = self.memory.repeat(input_shape[0], 1, 1)
    #     return memory
    
    # def pad_attention_mask(self, attention_mask, shape):
    #     if self.num_mem_tokens in {0, None}:
    #         return attention_mask
    #     else:
    #         mask = torch.ones(*shape[:2]).to(dtype=attention_mask.dtype, device=attention_mask.device)
    #         mask[:, self.num_mem_tokens:-self.num_mem_tokens] = attention_mask
    #         return mask

    # def pad_position_ids(self, position_ids, shape):
    #     if self.num_mem_tokens in {0, None}:
    #         return position_ids
    #     else:
    #         positions = torch.zeros(*shape[:2]).to(dtype=position_ids.dtype, device=position_ids.device)
    #         positions[:, self.num_mem_tokens:-self.num_mem_tokens] = position_ids
    #         return positions

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        X: Optional[torch.FloatTensor] = None,
        X_modalities: Optional[List[str]] = None,
        X_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        memory_state=None
    ) -> Union[Tuple, CausalLMOutputWithPast]:


        # print("=========forward method before ===========")
        # print(self.model.image_tower.image_tower.dtype)
        # print("====================")

        if inputs_embeds is None:
            (
                seg_input_ids,
                seg_position_ids,
                seg_attention_mask,
                past_key_values,
                seg_inputs_embeds,
                seg_labels
            ) = self.prepare_retro_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                X,
                X_sizes,
                X_modalities
            )

        num_segments =  len(seg_inputs_embeds)
        seg_logits = []
        for i in range(num_segments):
            input_ids = seg_input_ids
            position_ids = seg_position_ids[i]
            attention_mask = seg_attention_mask[i]
            inputs_embeds = seg_inputs_embeds[i]

            # if memory_state is None:
            #     memory_state = self.set_memory(inputs_embeds.shape)

            # # inputs: add memory
            # if self.num_mem_tokens in {0, None}:
            #     inputs_embeds = inputs_embeds
            # else:
            #     inputs_embeds = torch.cat([memory_state, inputs_embeds, memory_state], dim=1)
            # if attention_mask is not None:
            #     attention_mask = self.pad_attention_mask(attention_mask, inputs_embeds.shape)
            # if position_ids is not None:
            #     position_ids = self.pad_position_ids(position_ids, inputs_embeds.shape)
            
            inputs_embeds = inputs_embeds

            model_outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                # position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=True
            )   

            # # outputs: remove memory
            # if self.num_mem_tokens not in {0, None}:
            #     outputs = CausalLMOutputWithPast()
            #     memory_state = model_outputs.hidden_states[-1][:, -self.num_mem_tokens:]
            #     outputs["logits"] = model_outputs.logits[:, self.num_mem_tokens:-self.num_mem_tokens]

            #     if output_hidden_states is not None:
            #         outputs["hidden_states"] = [lh[:, self.num_mem_tokens:-self.num_mem_tokens] for lh in model_outputs.hidden_states]
            #     if output_attentions is not None:
            #         outputs["attentions"] = model_outputs["attentions"]

            #     seg_logits.append(model_outputs.logits[:, self.num_mem_tokens:-self.num_mem_tokens])
            # else:
            #     memory_state = None
            #     outputs = model_outputs
            #     seg_logits.append(outputs.logits)
            
            memory_state = None
            outputs = model_outputs
            seg_logits.append(outputs.logits)

            # # TODO: memory management
            # if i != 0:
            #     memory_state = memory_state.detach()
        
        # process output
        logits = torch.cat(seg_logits, dim=1)
        # print(torch.cat(seg_inputs_embeds, dim=1).shape)
        # print(logits.shape)
        # print(seg_labels.shape)
        loss = None
        # print(seg_labels)
        # print(seg_labels.shape)
        if seg_labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = seg_labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        X: Optional[torch.Tensor] = None,
        X_modalities: Optional[List[str]] = None,
        X_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if X is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                X,
                X_sizes,
                X_modalities,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        # print("debug="*20)
        # print(inputs_embeds.shape)
        # print("="*20)
        

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        X = kwargs.pop("X", None)
        X_sizes = kwargs.pop("X_sizes", None)
        X_modalities = kwargs.pop("X_modalities", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if X is not None:
            inputs['X'] = X
        if X_sizes is not None:
            inputs['X_sizes'] = X_sizes
        if X_modalities is not None:
            inputs['X_modalities'] = X_modalities
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLMRMT)
