import math

import torch


class SaveOutput:
    def __init__(self, register_outputs=True, register_inputs=False):
        self._handles = {}
        self.outputs = {}
        self.inputs = {}
        self.register_outputs = register_outputs
        self.register_inputs = register_inputs

    def register(self, module, module_name):
        module.module_name = module_name
        self._handles[module_name] = module.register_forward_hook(self.__call__)

    def __call__(self, module, module_in, module_out):
        if not hasattr(module, 'module_name'):
            raise AttributeError('All modules should have name attr')
        if self.register_outputs:
            self.outputs[module.module_name] = module_out.clone().detach()
        if self.register_inputs:
            self.inputs[module.module_name] = module_in.clone().detach()

    def unregister(self):
        for handle in self._handles.values():
            handle.remove()

    def clear(self):
        self.outputs = {}
        self.inputs = {}


def get_attention_map(unet, save_output, word_id, id_in_subbatch, subbatch_size=3, final_size=(64, 64)):
    amaps = []
    for name, layer in unet.named_modules():
        if name.endswith('attn2'):
            query = save_output.outputs[name + '.to_q']
            key = save_output.outputs[name + '.to_k']

            attn_heads = layer.heads
            _, k_seq_len, _ = key.shape
            batch_size, q_seq_len, inner_dim = query.shape

            head_dim = inner_dim // attn_heads

            # Separate heads: (batch_size, seq_len, attn_heads * head_dim) -> (batch_size, attn_heads, seq_len, head_dim)
            query = query.view(batch_size, -1, attn_heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn_heads, head_dim).transpose(1, 2)

            # Calculate attention maps: (batch_size, attn_heads, q_seq_len, k_seq_len)
            scale_factor = 1 / math.sqrt(query.size(-1))
            attn_weight = query @ key.transpose(-2, -1) * scale_factor
            attn_weight = torch.softmax(attn_weight, dim=-1)

            map_size = int(math.sqrt(q_seq_len))

            # Separate subbatches: (subbatch_size, batch_size // subbatch_size, attn_heads, q_seq_len, k_seq_len)
            attn_weight = attn_weight.view(subbatch_size, batch_size // subbatch_size, attn_heads, q_seq_len, k_seq_len)
            # Get only one map in each subbatch: (batch_size // subbatch_size, attn_heads, q_seq_len, k_seq_len)
            target_maps = attn_weight[id_in_subbatch]
            # Aggregate over all heads and take only specific key: (batch_size // subbatch_size, map_size, map_size)
            heat_maps = target_maps.mean(1)[..., word_id].reshape(-1, map_size, map_size)
            # Resize heat maps: (1, batch_size // subbatch_size, final_size[0], final_size[1])
            amaps.append(torch.nn.functional.interpolate(heat_maps[None], size=final_size, mode='bilinear'))

    # Aggregate over all layers: (batch_size // subbatch_size, final_size[0], final_size[1])
    amap = torch.cat(amaps).mean(0)
    return amap


def count_trainable_params(module, verbose=False):
    numel = 0
    for name, param in module.named_parameters():
        if param.requires_grad:
            numel += torch.numel(param)
            if verbose:
                print(name, torch.numel(param), sep='\t\t')

    return numel


@torch.no_grad()
def params_grad_norm(parameters):
    grad_norm_squared = 0
    for param in parameters:
        if param.grad is not None:
            grad_norm_squared += torch.square(torch.linalg.norm(param.grad)).item()

    return grad_norm_squared ** 0.5
