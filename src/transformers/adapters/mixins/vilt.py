import logging
from typing import Iterable, Tuple

import torch
import torch.nn as nn

from ..layer import AdapterLayer
from ..context import AdapterSetup, ForwardContext
from ..composition import AdapterCompositionBlock, BatchSplit, Fuse, Parallel, Split, Stack
from ..model_mixin import InvertibleAdaptersMixin, ModelAdaptersMixin


logger = logging.getLogger(__name__)


# For backwards compatibility, BertSelfOutput inherits directly from AdapterLayer
class ViltSelfOutputAdaptersMixin(AdapterLayer):
    """Adds adapters to the BertSelfOutput module."""

    def __init__(self):
        super().__init__("mh_adapter", None)

    def adapter_layer_forward(self, hidden_states, input_tensor, layer_norm):
        """
        Called for each forward pass through adapters.
        """
        if getattr(self.config, "is_adaptable", False):
            # First check current context before falling back to defined setup
            context = AdapterSetup.get_context()
            if context is not None:
                adapter_setup = context.adapter_setup
            else:
                adapter_setup = self.config.adapters.active_setup
        else:
            adapter_setup = None
        skip_adapters = adapter_setup is None or (
            self.config.adapters.skip_layers is not None and self.layer_idx in self.config.adapters.skip_layers
        )
        if not skip_adapters and (len(set(self.adapters.keys()) & adapter_setup.flatten()) > 0):
            input_hidden_states = hidden_states

            if isinstance(adapter_setup, Stack):
                hidden_states, _, input_tensor = self.adapter_stack(
                    adapter_setup, hidden_states, input_tensor, layer_norm
                )
            elif isinstance(adapter_setup, Fuse):
                hidden_states = self.adapter_fusion(adapter_setup, hidden_states, input_tensor, layer_norm)
            elif isinstance(adapter_setup, Split):
                hidden_states = self.adapter_split(adapter_setup, hidden_states, input_tensor, layer_norm)
            elif isinstance(adapter_setup, Parallel):
                # notice that we are overriding input tensor here to keep the same dim as hidden_states for the residual
                # in case we were blowing up the batch for parallel processing of multiple adapters for the same input
                hidden_states, input_tensor = self.adapter_parallel(
                    adapter_setup, hidden_states, input_tensor, layer_norm
                )
            elif isinstance(adapter_setup, BatchSplit):
                hidden_states = self.adapter_batchsplit(adapter_setup, hidden_states, input_tensor, layer_norm)
            else:
                raise ValueError(f"Invalid adapter setup {adapter_setup}")

            last_adapter = self.adapters[adapter_setup.last()]
            hidden_states = last_adapter.post_forward(hidden_states, input_hidden_states, input_tensor, layer_norm)

        elif layer_norm:
            hidden_states = layer_norm(hidden_states + input_tensor)
        elif input_tensor is not None:
            hidden_states = hidden_states + input_tensor

        return hidden_states


# For backwards compatibility, BertOutput inherits directly from AdapterLayer
class ViltOutputAdaptersMixin(AdapterLayer):
    """Adds adapters to the BertOutput module."""

    def __init__(self):
        super().__init__("output_adapter", None)

    def adapter_layer_forward(self, hidden_states, input_tensor, layer_norm):
        """
        Called for each forward pass through adapters.
        """
        if getattr(self.config, "is_adaptable", False):
            # First check current context before falling back to defined setup
            context = AdapterSetup.get_context()
            if context is not None:
                adapter_setup = context.adapter_setup
            else:
                adapter_setup = self.config.adapters.active_setup
        else:
            adapter_setup = None
        skip_adapters = adapter_setup is None or (
            self.config.adapters.skip_layers is not None and self.layer_idx in self.config.adapters.skip_layers
        )
        if not skip_adapters and (len(set(self.adapters.keys()) & adapter_setup.flatten()) > 0):
            input_hidden_states = hidden_states

            if isinstance(adapter_setup, Stack):
                hidden_states, _, input_tensor = self.adapter_stack(
                    adapter_setup, hidden_states, input_tensor, layer_norm
                )
            elif isinstance(adapter_setup, Fuse):
                hidden_states = self.adapter_fusion(adapter_setup, hidden_states, input_tensor, layer_norm)
            elif isinstance(adapter_setup, Split):
                hidden_states = self.adapter_split(adapter_setup, hidden_states, input_tensor, layer_norm)
            elif isinstance(adapter_setup, Parallel):
                # notice that we are overriding input tensor here to keep the same dim as hidden_states for the residual
                # in case we were blowing up the batch for parallel processing of multiple adapters for the same input
                hidden_states, input_tensor = self.adapter_parallel(
                    adapter_setup, hidden_states, input_tensor, layer_norm
                )
            elif isinstance(adapter_setup, BatchSplit):
                hidden_states = self.adapter_batchsplit(adapter_setup, hidden_states, input_tensor, layer_norm)
            else:
                raise ValueError(f"Invalid adapter setup {adapter_setup}")

            last_adapter = self.adapters[adapter_setup.last()]
            hidden_states = last_adapter.post_forward(hidden_states, input_hidden_states, input_tensor, layer_norm)

        elif layer_norm:
            hidden_states = layer_norm(hidden_states + input_tensor)
        elif input_tensor is not None:
            hidden_states = hidden_states + input_tensor

        return hidden_states


class ViltModelAdaptersMixin(ModelAdaptersMixin):
    """Adds adapters to the BertModel module."""

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.encoder.layer):
            yield i, layer