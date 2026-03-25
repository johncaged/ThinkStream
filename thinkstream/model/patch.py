from liger_kernel.transformers.model import loss_utils

# The original loss_utils
LigerForCausalLMLoss = loss_utils.LigerForCausalLMLoss


def _LigerForCausalLMLoss(*args, **kwargs):
    # Remove related kwargs to avoid error
    # NOTE: video_block_mask is for flex attention
    kwargs.pop("video_block_mask")
    return LigerForCausalLMLoss(*args, **kwargs)


loss_utils.LigerForCausalLMLoss = _LigerForCausalLMLoss


# NOTE: We should patch LigerForCausalLMLoss first, and then patch the following models
from transformers.utils import can_return_tuple
from liger_kernel.transformers.model import qwen2_5_vl
from liger_kernel.transformers.model import qwen3_vl

lce_forward_qwen2_5_vl = qwen2_5_vl.lce_forward
lce_forward_qwen3_vl = qwen3_vl.lce_forward

from thinkstream.model.streaming_attention import (
    create_mask,
    generate_video_sliding_window_mask_mod,
)
from thinkstream.model import DEFAULT_VIDEO_FLEX_WINDOW_SIZE


def build_video_block_mask(model, video_mask, attention_mask):
    """Create ``video_block_mask`` from ``video_mask`` and ``attention_mask``.

    This is the shared helper used by all patched forwards (SFT / GRPO) so
    that the sliding-window flex-attention mask is built in exactly one place.

    Returns ``None`` when ``video_mask`` is ``None``.
    """
    if video_mask is None:
        return None
    assert attention_mask is not None
    window_size_n = getattr(
        model.config, "video_flex_window_size", DEFAULT_VIDEO_FLEX_WINDOW_SIZE
    )
    B, L = video_mask.shape
    assert video_mask.shape == attention_mask.shape
    mask_mod = generate_video_sliding_window_mask_mod(
        video_mask.contiguous(), attention_mask.contiguous(), window_size_n
    )
    return create_mask(
        mask_mod,
        model.training,
        B=B,
        H=None,
        Q_LEN=L,
        KV_LEN=L,
        device=model.device,
    )


@can_return_tuple
def _lce_forward_qwen2_5_vl(
    self,
    *args,
    attention_mask=None,
    video_mask=None,
    **kwargs,
):
    video_block_mask = build_video_block_mask(self, video_mask, attention_mask)
    return lce_forward_qwen2_5_vl(
        self,
        *args,
        attention_mask=attention_mask,
        video_block_mask=video_block_mask,
        **kwargs,
    )


@can_return_tuple
def _lce_forward_qwen3_vl(
    self,
    *args,
    attention_mask=None,
    video_mask=None,
    **kwargs,
):
    video_block_mask = build_video_block_mask(self, video_mask, attention_mask)
    return lce_forward_qwen3_vl(
        self,
        *args,
        attention_mask=attention_mask,
        video_block_mask=video_block_mask,
        **kwargs,
    )


qwen2_5_vl.lce_forward = _lce_forward_qwen2_5_vl
qwen3_vl.lce_forward = _lce_forward_qwen3_vl
print("Successfully Patched!")


# ---------------------------------------------------------------------------
# Liger fused-linear GRPO forward  (replaces lce_forward for GRPO training)
# ---------------------------------------------------------------------------

import torch
from liger_kernel.chunked_loss.grpo_loss import LigerFusedLinearGRPOLoss

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLCausalLMOutputWithPast,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast,
)


def _pack_by_completion_mask(
    shifted_hs,
    shifted_labels,
    shifted_loss_mask,
    old_per_token_logps,
    ref_input=None,
):
    """Pack sequences to [B, max_valid_len, ...] so only valid (assistant) positions remain.

    Removes gaps from the middle (e.g. video-only segments). Mask is 1 for valid
    and 0 only for padding at the end, so no chunk is fully masked in the middle.
    Returns (packed_hs, packed_labels, packed_mask, packed_old, packed_ref_input),
    or None if max_valid_len == 0. advantages are not packed; pass [B] directly to the loss.
    """
    B, L, H = shifted_hs.shape
    device = shifted_hs.device
    dtype_hs = shifted_hs.dtype

    valid_lens = shifted_loss_mask.sum(dim=1).long()  # [B]
    max_valid_len = int(valid_lens.max().item())
    if max_valid_len == 0:
        return None

    # Build packed tensors: for each b, take positions where mask[b]==1, then pad to max_valid_len
    packed_hs = torch.zeros(B, max_valid_len, H, device=device, dtype=dtype_hs)
    packed_labels = torch.zeros(
        B, max_valid_len, device=device, dtype=shifted_labels.dtype
    )
    packed_mask = torch.zeros(
        B, max_valid_len, device=device, dtype=shifted_loss_mask.dtype
    )

    if old_per_token_logps is not None:
        packed_old = torch.zeros(
            B, max_valid_len, device=device, dtype=old_per_token_logps.dtype
        )
    else:
        packed_old = None

    if ref_input is not None:
        packed_ref_input = torch.zeros(
            B, max_valid_len, H, device=device, dtype=ref_input.dtype
        )
    else:
        packed_ref_input = None

    for b in range(B):
        indices = (shifted_loss_mask[b] == 1).nonzero(as_tuple=True)[0]
        n = indices.numel()
        if n == 0:
            continue
        packed_hs[b, :n] = shifted_hs[b, indices]
        packed_labels[b, :n] = shifted_labels[b, indices]
        packed_mask[b, :n] = 1
        if old_per_token_logps is not None:
            packed_old[b, :n] = old_per_token_logps[b, indices]
        if ref_input is not None:
            packed_ref_input[b, :n] = ref_input[b, indices]

    return (
        packed_hs,
        packed_labels,
        packed_mask,
        packed_old,
        packed_ref_input,
    )


def _grpo_lce_forward_common(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    rope_deltas=None,
    cache_position=None,
    second_per_grid_ts=None,
    logits_to_keep=0,
    advantages=None,
    old_per_token_logps=None,
    ref_input=None,
    ref_weight=None,
    ref_bias=None,
    completion_mask=None,
    grpo_beta=0.04,
    grpo_loss_type="grpo",
    video_mask=None,
    output_cls=None,
    add_second_per_grid_ts=False,
    **kwargs,
):
    """Shared GRPO LCE forward logic. Call from model-specific forward with
    output_cls and add_second_per_grid_ts set (e.g. Qwen2.5-VL True, Qwen3-VL False).
    """
    video_block_mask = build_video_block_mask(self, video_mask, attention_mask)

    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    backbone_kwargs = dict(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        video_block_mask=video_block_mask,
        **kwargs,
    )
    if add_second_per_grid_ts:
        backbone_kwargs["second_per_grid_ts"] = second_per_grid_ts
    outputs = self.model(**backbone_kwargs)
    hidden_states = outputs[0]

    loss = None
    logits = None

    if advantages is not None:
        assert completion_mask is not None
        shifted_hs = hidden_states[:, :-1, :].contiguous()
        shifted_labels = input_ids[:, 1:]
        shifted_loss_mask = completion_mask[:, 1:]
        adv = advantages.to(shifted_hs.device)

        packed = _pack_by_completion_mask(
            shifted_hs,
            shifted_labels,
            shifted_loss_mask,
            old_per_token_logps,
            ref_input=ref_input,
        )
        if packed is None:
            loss = shifted_hs.new_zeros([])
        else:
            packed_hs, packed_labels, packed_mask, packed_old, packed_ref_input = packed
            grpo_loss_fn = LigerFusedLinearGRPOLoss(
                beta=grpo_beta,
                loss_type=grpo_loss_type,
                use_ref_model=True,
                compiled=True,
            )
            result = grpo_loss_fn(
                _input=packed_hs,
                lin_weight=self.lm_head.weight,
                selected_token_ids=packed_labels,
                attention_mask=packed_mask,
                advantages=adv,
                bias=self.lm_head.bias,
                old_per_token_logps=packed_old,
                ref_input=packed_ref_input,
                ref_weight=ref_weight,
                ref_bias=ref_bias,
            )
            loss = result[0] if isinstance(result, tuple) else result
    else:
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if labels is not None:
            text_config = getattr(self.config, "text_config", self.config)
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=text_config.vocab_size,
            )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return output_cls(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=getattr(outputs, "rope_deltas", None),
    )


def grpo_lce_forward_qwen2_5_vl(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    rope_deltas=None,
    cache_position=None,
    second_per_grid_ts=None,
    logits_to_keep=0,
    advantages=None,
    old_per_token_logps=None,
    ref_input=None,
    ref_weight=None,
    ref_bias=None,
    completion_mask=None,
    grpo_beta=0.04,
    grpo_loss_type="grpo",
    video_mask=None,
    **kwargs,
):
    """GRPO-aware forward for Qwen2.5-VL (uses LigerFusedLinearGRPOLoss when advantages given)."""
    return _grpo_lce_forward_common(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        rope_deltas=rope_deltas,
        cache_position=cache_position,
        second_per_grid_ts=second_per_grid_ts,
        logits_to_keep=logits_to_keep,
        advantages=advantages,
        old_per_token_logps=old_per_token_logps,
        ref_input=ref_input,
        ref_weight=ref_weight,
        ref_bias=ref_bias,
        completion_mask=completion_mask,
        grpo_beta=grpo_beta,
        grpo_loss_type=grpo_loss_type,
        video_mask=video_mask,
        output_cls=Qwen2_5_VLCausalLMOutputWithPast,
        add_second_per_grid_ts=True,
        **kwargs,
    )


def grpo_lce_forward_qwen3vl(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    pixel_values=None,
    pixel_values_videos=None,
    image_grid_thw=None,
    video_grid_thw=None,
    rope_deltas=None,
    cache_position=None,
    second_per_grid_ts=None,
    logits_to_keep=0,
    advantages=None,
    old_per_token_logps=None,
    ref_input=None,
    ref_weight=None,
    ref_bias=None,
    completion_mask=None,
    grpo_beta=0.04,
    grpo_loss_type="grpo",
    video_mask=None,
    **kwargs,
):
    """GRPO-aware forward for Qwen3-VL (uses LigerFusedLinearGRPOLoss when advantages given)."""
    return _grpo_lce_forward_common(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        rope_deltas=rope_deltas,
        cache_position=cache_position,
        second_per_grid_ts=second_per_grid_ts,
        logits_to_keep=logits_to_keep,
        advantages=advantages,
        old_per_token_logps=old_per_token_logps,
        ref_input=ref_input,
        ref_weight=ref_weight,
        ref_bias=ref_bias,
        completion_mask=completion_mask,
        grpo_beta=grpo_beta,
        grpo_loss_type=grpo_loss_type,
        video_mask=video_mask,
        output_cls=Qwen3VLCausalLMOutputWithPast,
        add_second_per_grid_ts=False,
        **kwargs,
    )


# model_type -> GRPO LCE forward (same keys as thinkstream.model.MODEL_CLS)
GRPO_LCE_FORWARD = {
    "qwen2.5vl": grpo_lce_forward_qwen2_5_vl,
    "qwen3vl": grpo_lce_forward_qwen3vl,
}
