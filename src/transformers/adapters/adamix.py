import math
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configuration_utils import PretrainedConfig
from .composition import AdapterCompositionBlock
from .configuration import AdaMixConfig, LoRAConfig
from .layer import AdapterLayerBase
from .lora import LoRALayer, Linear, LoRA
import copy

from typing import Union, Optional


class AdaMix(LoRA):

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

    def add_adapter(self, adapter_name: str, layer_idx: int) -> bool:
        pass


class AdaMixLayer(AdapterLayerBase, AdaMix):
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
            adamix = AdaMix(*self._get_lora_shapes(adamix_config), adamix_config)
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
        if self.merged:
            pass
        else:
            pass
