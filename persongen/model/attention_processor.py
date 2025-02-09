from typing import Optional

import torch

from diffusers.models.attention_processor import Attention, AttnProcessor


class AttnProcessorPhotoswap(AttnProcessor):
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        num_images_per_prompt=1,
        change_self_attn_map=False, change_self_attn_feat=False, change_cross_attn_map=False
    ) -> torch.Tensor:
        is_cross_attn = encoder_hidden_states is not None

        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        batch_size, sequence_length, inner_dim = hidden_states.shape

        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)    # (batch_size, k_seq_len, dim)
        value = attn.to_v(encoder_hidden_states)  # (batch_size, k_seq_len, dim)

        query = attn.head_to_batch_dim(query)  # (batch_size * head_size, q_seq_len, dim // head_size)
        key = attn.head_to_batch_dim(key)      # (batch_size * head_size, k_seq_len, dim // head_size)
        value = attn.head_to_batch_dim(value)  # (batch_size * head_size, k_seq_len, dim // head_size)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        attention_probs = attn.batch_to_head_dim(attention_probs)
        if is_cross_attn:
            if change_cross_attn_map:
                attention_probs[1 * num_images_per_prompt: 2 * num_images_per_prompt] = attention_probs[3 * num_images_per_prompt: 4 * num_images_per_prompt].clone()
        else:
            if change_self_attn_map:
                attention_probs[1 * num_images_per_prompt: 2 * num_images_per_prompt] = attention_probs[3 * num_images_per_prompt: 4 * num_images_per_prompt].clone()

        attention_probs = attn.head_to_batch_dim(attention_probs)

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        if not is_cross_attn and change_self_attn_feat:
            hidden_states[1 * num_images_per_prompt: 2 * num_images_per_prompt] = hidden_states[3 * num_images_per_prompt: 4 * num_images_per_prompt].clone()

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
