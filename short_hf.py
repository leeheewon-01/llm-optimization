from typing import List, Optional, Dict
import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

def block_influence(
    input_hidden_state: torch.Tensor,
    output_hidden_state: torch.Tensor,
    angular=False,
):
    """
    input_hidden_state: B, S, D
    output_hidden_state: B, S, D
    """
    _, _, d = input_hidden_state.shape
    input_hidden_state = input_hidden_state.reshape(-1, d)
    output_hidden_state = output_hidden_state.reshape(-1, d)

    norm_input = input_hidden_state.norm(dim=-1, keepdim=True)
    norm_output = output_hidden_state.norm(dim=-1, keepdim=True)

    sim = (input_hidden_state @ output_hidden_state.T) / (norm_input * norm_output)
    sim = sim.diagonal().nan_to_num(nan=0.5)

    if angular:
        return (torch.arccos(sim) / torch.pi)

    return 1 - sim

from repeat_model import AutoLayerRepeatingForCausalLM

class ShortHFModel():

    def __init__(self, model_name: str, layers_path: str, layer_repeat_config: Optional[Dict[int, int]] = None):
        """
        HuggingFace Model Wrapper

        Args:
            model_name (str): HuggingFace model name
            layers_path (str): String in dot notation demonstrating how to access layers of the model. Ex: "model.layers"
            layer_repeat_config (Dict[int, int]): Dict mapping layer indices to number of repetitions
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Use the custom from_pretrained method
        self.model = AutoLayerRepeatingForCausalLM.from_pretrained(
            model_name, 
            layer_repeat_config=layer_repeat_config,
            torch_dtype=torch.float16
        )
        self.model.to("cuda")

        # Access layers through the custom model's layers attribute
        self.layers = self.model.layers
        
        # n_prune_layers is calculated as 0.271 * num_hidden_layers
        self.n_prune_layers = int(self.model.config.num_hidden_layers * 0.271)
        self.importances = [0 for _ in self.layers]  # layer-wise importance scores

        print(f"n_prune_layers: {self.n_prune_layers}")

    def remove_layers(
        self,
        layers_to_remove: Optional[List[int]] = [],
        angular: Optional[bool] = False
    ):
        if angular:
            assert self.importances, "Need to compute importances with eval_importance()"
            assert self.n_prune_layers, "Need number of layers to prune, set `n_prune_layers`"
            start_layer = np.argsort(np.array(self.importances[:-self.n_prune_layers+1]))[0]
            layers_to_remove = list(range(start_layer, start_layer + self.n_prune_layers))
        elif not layers_to_remove and self.n_prune_layers:
            assert self.importances, "Need to compute importances with eval_importance()"
            layers_to_remove = np.argsort(np.array(self.importances))[:self.n_prune_layers].tolist()

        # remove layers in reverse to avoid indexing errors
        for layer_idx in sorted(layers_to_remove, reverse=True):
            try:
                del self.layers[layer_idx]
            except IndexError:
                print(f"layer {layer_idx} does not exist, function may have already been called")
                return []
        
        return layers_to_remove
    
    def compute_bi(self, hiddens: List[torch.Tensor], angular: bool):
        n = 1
        if angular:
            assert self.n_prune_layers is not None, "Set number of layers to prune to use angular importance"
            n = self.n_prune_layers

        for i in range(len(hiddens) - n):
            in_hidden = hiddens[i]
            out_hidden = hiddens[i+n]
            if angular:
                # use only last token for angular distance as described in section 3.2
                # https://arxiv.org/pdf/2403.17887.pdf
                in_hidden = in_hidden[:,-1:]
                out_hidden = out_hidden[:,-1:]
            
            self.importances[i] += block_influence(
                in_hidden,
                out_hidden,
                angular=angular
            ).sum().cpu().item()

    @torch.inference_mode()
    def eval_importance(
        self,
        prompts: List[str],
        max_seq_len: int,
        stride: int = 256,
        max_gen_len: int = 0,
        temperature: float = 0.6,
        top_p: float = 0.9,
        angular: Optional[bool] = False
    ):
        """
        Computes layer-wise importances over input texts.

        NOTE: ShortGPT paper performs no generation during importance computation, which suggests a `max_gen_len`= 0.

        Args:
            prompts (List[str]): List of prompts.
            max_seq_len (int): Maximum sequence length for model input, the sliding window size.
            (Optional) stride (int): Number of tokens to skip/shift between each window inference.
            (Optional) max_gen_len (int): Maximum length of the generated text sequence.
            (Optional) temperature (float): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            (Optional) top_p (float): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            (Optional) angular (bool): Whether to use angular distance. Defaults to False.

        Returns:
            None
        """
        prompt_tokens = self.tokenizer(
            prompts,
            padding=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids = prompt_tokens.input_ids
        attn_mask = prompt_tokens.attention_mask

        max_prompt_len = max(len(t) for t in input_ids)

        # authors use a sliding window of size 1024 with a shift of 256
        for start in range(0, max_prompt_len, stride):
            seq_ids = (attn_mask.sum(dim=-1) > start).nonzero().squeeze()
            seq_ids = seq_ids.unsqueeze(0) if seq_ids.dim() == 0 else seq_ids  # ensure 2d
            inputs = input_ids[seq_ids, start:start+max_seq_len]
            attn = attn_mask[seq_ids, start:start+max_seq_len]

            if max_gen_len == 0:
                # When not generating, don't pass generation-specific parameters
                outputs = self.model(
                    input_ids=inputs.to("cuda"),
                    attention_mask=attn.to("cuda"),
                    output_hidden_states=True,
                )
                # Extract hidden states from the base model output
                if hasattr(outputs, 'hidden_states'):
                    hidden_states = outputs.hidden_states
                else:
                    # If it's a CausalLMOutput, we need to wrap hidden_states properly
                    hidden_states = outputs.hidden_states if outputs.hidden_states else []
            else:
                outputs = self.model.generate(
                    input_ids=inputs.to("cuda"),
                    attention_mask=attn.to("cuda"),
                    max_new_tokens=max_gen_len, 
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )
                hidden_states = outputs.hidden_states
            
            self.compute_bi(hidden_states, angular=angular)

        return