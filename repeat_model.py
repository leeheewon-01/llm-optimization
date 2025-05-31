import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoModel, AutoModelForCausalLM, AutoConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
import copy


class LayerRepeatingWrapper(nn.ModuleList):
    """
    A wrapper that intercepts layer calls to implement layer repetition.
    """
    def __init__(self, layers, layer_repeat_config: Dict[int, int] = None):
        super().__init__(layers)
        self.layer_repeat_config = layer_repeat_config or {}
        self._original_len = len(layers)
        
    def __iter__(self):
        for idx in range(self._original_len):
            repeat_count = self.layer_repeat_config.get(idx, 1)
            for _ in range(repeat_count):
                yield self[idx]
                
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Handle slicing by returning a regular ModuleList
            return nn.ModuleList(list(self._modules.values())[idx])
        else:
            return self._modules[self._get_abs_string_index(idx)]


class AutoLayerRepeatingForCausalLM(nn.Module):
    """
    Custom CausalLM that repeats specific layers during inference.
    """
    
    def __init__(self, base_model, layer_repeat_config: Dict[int, int]):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config
        self.layer_repeat_config = layer_repeat_config
        
        # Store the original layers before wrapping
        if hasattr(base_model.model, 'layers'):
            # Keep a reference to the original ModuleList
            self._original_layers = base_model.model.layers
            # Create and assign the wrapper
            self.wrapped_layers = LayerRepeatingWrapper(base_model.model.layers, layer_repeat_config)
            base_model.model.layers = self.wrapped_layers
        else:
            raise ValueError("Model doesn't have 'model.layers' attribute")
            
        # Copy other necessary attributes
        self.vocab_size = base_model.vocab_size
    
    @property
    def layers(self):
        """Access to the wrapped layers for compatibility."""
        return self.wrapped_layers
        
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, layer_repeat_config: Dict[int, int], **kwargs):
        """
        Load a pretrained model and wrap it with layer repeating functionality.
        """
        # Remove layer_repeat_config from kwargs to avoid passing it to base model
        kwargs.pop('layer_repeat_config', None)
        
        # Load the base model
        base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)
        
        # Create the custom model
        return cls(base_model, layer_repeat_config)
    
    @property
    def model(self):
        """Direct access to the base model's model component."""
        return self.base_model.model
    
    @property
    def lm_head(self):
        """Direct access to the base model's lm_head."""
        return self.base_model.lm_head
    
    def forward(self, *args, **kwargs):
        """Forward all calls to the base model."""
        return self.base_model(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        """Forward generate calls to the base model."""
        return self.base_model.generate(*args, **kwargs)
    
    def to(self, *args, **kwargs):
        """Ensure all components are moved to the same device."""
        self.base_model = self.base_model.to(*args, **kwargs)
        return self
    
    def __getattr__(self, name):
        """Forward any other attribute access to the base model."""
        if name in ['base_model', 'config', 'layer_repeat_config', 'original_layers', 'wrapped_layers']:
            return super().__getattr__(name)
        return getattr(self.base_model, name)