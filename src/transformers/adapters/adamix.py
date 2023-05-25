import math
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configuration_utils import PretrainedConfig
from .composition import AdapterCompositionBlock
from .configuration import AdaMixConfig, LoRAConfig
from .layer import AdapterLayerBase
from .lora import LoRA, Linear
import copy

from typing import Union, Optional


class Nope(LoRA):

    def __int__(self,
                in_features: int,
                out_features: int,
                location_key: str,
                config: Union[PretrainedConfig, AdaMixConfig],
                attn_key: str = None,
                fan_in_fan_out: bool = False,
                no_init_bias: bool = False,
                **kwargs):
        # TODO: Build one "mixable" Lora Layer. Should make strong use of existing Lora implementation
        config_dict = config.to_dict()
        config_dict['architecture'] = 'lora'
        lora_config = LoRAConfig(**config_dict)
        lora = LoRA(
            *self._get_lora_shapes(lora_config),
            lora_config,
            gating_heads=self.get_n_heads(lora_config),
        )

        if config.share_A:
            self.experts_lora_A = torch.nn.ParameterList([copy.deepcopy(lora.lora_A) for _ in range(1)])
        else:
            self.experts_lora_A = torch.nn.ParameterList([copy.deepcopy(lora.lora_A) for _ in range(config.adaption_modules)])
        if config.share_B:
            self.experts_lora_B = torch.nn.ParameterList([copy.deepcopy(lora.lora_B) for _ in range(1)])
        else:
            self.experts_lora_B = torch.nn.ParameterList([copy.deepcopy(lora.lora_B) for _ in range(config.adaption_modules)])


        self.expert_score_weights = torch.nn.Parameter(torch.zeros(config.adaption_modules), requires_grad=False)  # Todo: figure out what this does



class AdaMixLayer(AdapterLayerBase):
    def __init__(self, location_key: str, config: PretrainedConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.location_key = location_key + "_lora_adamix"
        self.config = config
        self.lora_adaptions = nn.ModuleDict(dict())

        self.merged = False

    def get_n_heads(self, lora: Union[LoRA, AdaMixConfig]):
        return 1

    def _check_lora_location(self, config: AdaMixConfig):
        return True

    def _get_lora_shapes(self, config: AdaMixConfig):
        raise NotImplementedError()

    def add_adapter(self, adapter_name: str, layer_idx: int) -> bool:
        self.layer_idx = layer_idx
        adamix_config = self.config.adapters.match(
            adapter_name,
            config_type=AdaMixConfig,
            layer_idx=self.layer_idx,
            location_key=self.location_key,
        )
        if adamix_config is not None and self._check_lora_location(adamix_config):
            adamix = Nope(*self._get_lora_shapes(adamix_config), adamix_config)
            adamix.train(self.training)
            self.lora_adaptions[adapter_name] = adamix
            return True

        return False

    def delete_adapter(self, adapter_name: str):
        if adapter_name in self.lora_adaptions:
            del self.lora_adaptions[adapter_name]

    def add_fusion_layer(self, adapter_names: Union[List, str]):
        pass  # not applicable to lora

    def delete_fusion_layer(self, adapter_names: Union[List, str]):
        pass  # not applicable to lora

    def enable_adapters(self, adapter_setup: AdapterCompositionBlock, unfreeze_adapters: bool, unfreeze_fusion: bool):
        if unfreeze_adapters:
            for name in adapter_setup.flatten():
                if name in self.lora_adaptions:
                    for param in self.lora_adaptions[name].parameters():
                        param.requires_grad = True

    def get_adapter(self, adapter_name: str) -> Optional[nn.Module]:
        if adapter_name in self.lora_adaptions:
            return self.lora_adaptions[adapter_name]
        else:
            return

    def forward(self):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        def get_expert_id():
            # sample a random expert uniformly
            return torch.randint(low=0, high=self.config.adaption_modules, size=(1,)).item()
        if self.merged:
            # No need to sample
            pass
        else:
            if self.config.share_A:
                pass
            else:
                expert_idx = get_expert_id()
                expert_B = self.lora_adaptions
            if self.config.share_B:
                pass
            else:
                expert_id = get_expert_id()


class AdaMixLinear(Linear):
    def __init__(self,
            in_features: int,
            out_features: int,
            location_key: str,
            config: PretrainedConfig,
            attn_key: str = None,
            fan_in_fan_out: bool = False,
            no_init_bias: bool = False,
            **kwargs):
        super().__init__(in_features, out_features, location_key, config, attn_key, fan_in_fan_out, no_init_bias, **kwargs)

        share_A = share_B = num_experts = 0

        adapter_name = "adamix"

        adamix_config = self.config.adapters.match(
            adapter_name,
            config_type=AdaMixConfig,
            layer_idx=self.layer_idx,
            location_key=self.location_key,
        )
        if adamix_config is not None:
            print("Nay")
        if share_A == 1:
            self.experts_lora_A = torch.nn.ParameterList([copy.deepcopy(self.lora_A) for i in range(1)])
        else:
            self.experts_lora_A = torch.nn.ParameterList([copy.deepcopy(self.lora_A) for i in range(num_experts)])

        if share_B == 1:
            self.experts_lora_B = torch.nn.ParameterList([copy.deepcopy(self.lora_B) for i in range(1)])
        else:
            self.experts_lora_B = torch.nn.ParameterList([copy.deepcopy(self.lora_B) for i in range(num_experts)])

        self.share_A = share_A
        self.share_B = share_B

        # Remove original lora parameters
        self.lora_A = None
        self.lora_B = None

        self.num_experts = num_experts
        self.lora_expert_score_weight = torch.nn.Parameter(torch.zeros(self.num_experts), requires_grad=False)

        self.lora_A_w = None
        self.lora_B_w = None

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:

                if self.training and not self.lora_expert_score_weight.requires_grad:
                    if self.share_A == 1:
                        after_A = F.linear(self.lora_dropout(x), self.experts_lora_A[0])
                    else:
                        expert_idx = torch.randint(low=0, high=self.num_experts,
                                                   size=(1,)).item()  # selected expert
                        after_A = F.linear(self.lora_dropout(x), self.experts_lora_A[expert_idx])

                    if self.share_B == 1:
                        after_B = F.conv1d(
                            after_A.transpose(-2, -1),
                            self.experts_lora_B[0].unsqueeze(-1),
                            groups=sum(self.enable_lora)
                        ).transpose(-2, -1)
                    else:
                        expert_idx = torch.randint(low=0, high=self.num_experts,
                                                   size=(1,)).item()  # selected expert
                        after_B = F.conv1d(
                            after_A.transpose(-2, -1),
                            self.experts_lora_B[expert_idx].unsqueeze(-1),
                            groups=sum(self.enable_lora)
                        ).transpose(-2, -1)
                else:
                    expert_weights = F.softmax(self.lora_expert_score_weight, dim=-1)

                    if not self.training:
                        if self.lora_A_w is None and self.lora_B_w is None:
                            self.lora_A_w = 0.
                            self.lora_B_w = 0.

                            if self.share_A == 1:
                                self.lora_A_w = self.experts_lora_A[0]
                            else:
                                for idx in range(self.num_experts):
                                    self.lora_A_w += expert_weights[idx] * self.experts_lora_A[idx]

                            if self.share_B == 1:
                                self.lora_B_w = self.experts_lora_B[0]
                            else:
                                for idx in range(self.num_experts):
                                    self.lora_B_w += expert_weights[idx] * self.experts_lora_B[idx]

                        lora_A_w = self.lora_A_w
                        lora_B_w = self.lora_B_w

                    else:
                        lora_A_w = 0.
                        lora_B_w = 0.

                        if self.share_A == 1:
                            lora_A_w = self.experts_lora_A[0]
                        else:
                            for idx in range(self.num_experts):
                                lora_A_w += expert_weights[idx] * self.experts_lora_A[idx]

                        if self.share_B == 1:
                            lora_B_w = self.experts_lora_B[0]
                        else:
                            for idx in range(self.num_experts):
                                lora_B_w += expert_weights[idx] * self.experts_lora_B[idx]

                    after_A = F.linear(self.lora_dropout(x), lora_A_w)

                    after_B = F.conv1d(
                        after_A.transpose(-2, -1),
                        lora_B_w.unsqueeze(-1),
                        groups=sum(self.enable_lora)
                    ).transpose(-2, -1)

                result += self.zero_pad(after_B) * self.scaling
            return result
