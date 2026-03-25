import torch
from typing import List, Optional, Union, Any
from transformers.modeling_utils import AttentionInterface
from flash_attn import flash_attn_with_kvcache
from thinkstream.model import DEFAULT_VIDEO_FLEX_WINDOW_SIZE
from thinkstream.data.stream_data_processor import (
    DEFAULT_MAX_CHUNKS,
    DEFAULT_INFERENCE_MIN_PIXELS,
    DEFAULT_INFERENCE_MAX_PIXELS,
    FRAMES_PER_CHUNK as _FRAMES_PER_CHUNK,
    preload_video,
    compute_position_ids,
)

try:
    from flashinfer.sampling import top_k_top_p_sampling_from_logits

    FLASHINFER_AVAILABLE = True
    print("[INFO] Using flash infer for fast sampling.")
except ImportError:
    FLASHINFER_AVAILABLE = False
    print("[WARNING] Using PyTorch Fallback for sampling.")

    def top_k_top_p_sampling_from_logits(
        logits: torch.Tensor, top_k: int, top_p: float
    ) -> torch.Tensor:
        """
        Fallback implementation of fused sampling using native PyTorch.
        Matches the FlashInfer API signature for seamless integration.
        """
        # 1. Softmax
        probs = torch.softmax(logits, dim=-1)

        # 2. Top-K Filtering
        if top_k > 0:
            top_k_vals, _ = torch.topk(probs, top_k, dim=-1)
            min_val = top_k_vals[..., -1].unsqueeze(-1)
            probs = torch.where(
                probs < min_val, torch.tensor(0.0, device=probs.device), probs
            )

        # 3. Top-P (Nucleus) Filtering
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Ensure at least one token is kept
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=1, index=sorted_indices, src=sorted_indices_to_remove
            )
            probs = probs.masked_fill(indices_to_remove, 0.0)

        # 4. Renormalize & Sample
        probs_sum = torch.sum(probs, dim=-1, keepdim=True)
        probs = probs / (probs_sum + 1e-6)
        next_tokens = torch.multinomial(probs, num_samples=1)

        return next_tokens.squeeze(-1)  # Returns [Batch]


def flash_attention_2_infer(
    module: torch.nn.Module,
    query: torch.Tensor,  # [B, H, L, D]
    key: torch.Tensor,  # NOTE: Actually KV Cache
    value: torch.Tensor,  # NOTE: Actually KV Cache
    *args,
    attn_cache_seqlens: torch.Tensor,
    **kwargs,
):
    attn_output = flash_attn_with_kvcache(
        query.permute(0, 2, 1, 3),
        key.permute(0, 2, 1, 3),
        value.permute(0, 2, 1, 3),
        cache_seqlens=attn_cache_seqlens[module.layer_idx],
        causal=True,
        # NOTE: We keep num_splits=0 to enable heuristic.
    )
    return attn_output, None


AttentionInterface.register("flash_attention_2_infer", flash_attention_2_infer)


class StreamingCache:
    def __init__(
        self,
        batch_size: int,
        max_len: int,
        dtype: torch.dtype,
        device: torch.device,
        num_hidden_layers: int,
        num_key_value_heads: int,
        head_dim: int,
    ):
        self.batch_size = batch_size
        self.k_cache = []
        self.v_cache = []
        self.max_len = max_len
        self.num_hidden_layers = num_hidden_layers
        for _ in range(num_hidden_layers):
            self.k_cache.append(
                torch.zeros(
                    (batch_size, num_key_value_heads, max_len, head_dim),
                    dtype=dtype,
                    device=device,
                    requires_grad=False,
                )
            )
            self.v_cache.append(
                torch.zeros(
                    (batch_size, num_key_value_heads, max_len, head_dim),
                    dtype=dtype,
                    device=device,
                    requires_grad=False,
                )
            )
        # NOTE: The single source of truth for KV Cache Lengths per layer
        # Shape: [num_hidden_layers, batch_size]
        self.cache_seqlens = torch.zeros(
            (num_hidden_layers, batch_size), dtype=torch.int32, device=device
        )

    def adjust_seqlens(self, delta: torch.Tensor, layer_idx: Optional[int] = None):
        """
        Adjusts the sequence lengths by a delta.
        delta: Must be a Tensor of shape (batch_size,).
        If layer_idx is None, updates ALL layers (broadcasting).
        """
        # Strict checking for batch size consistency
        assert delta.shape[0] == self.batch_size, (
            f"Delta shape {delta.shape} mismatch with batch size {self.batch_size}"
        )
        if layer_idx is None:
            # Broadcast update to ALL layers
            self.cache_seqlens += delta.unsqueeze(0)
        else:
            # Update specific layer
            self.cache_seqlens[layer_idx] += delta

    def reset(self) -> None:
        """Resets the KV Cache and sequence lengths."""
        for layer in self.k_cache:
            layer.zero_()
        for layer in self.v_cache:
            layer.zero_()
        self.cache_seqlens.zero_()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, num_heads, seq_len, head_dim = key_states.shape
        assert bsz == self.batch_size, "Runtime batch size mismatch with Cache init."
        # Use layer-specific length
        # current_lengths shape: [Batch]
        current_lengths = self.cache_seqlens[layer_idx]
        # Calculate start indices based on current length
        # Prefill: 0 -> seq_len
        # Decode:  L -> L+1
        start_indices = current_lengths
        # Path 1: Decoding (Captured by CUDA Graph)
        if seq_len == 1:
            # Indices: [Batch, NumHeads, 1, HeadDim]
            indices = start_indices.view(bsz, 1, 1, 1).expand(
                bsz, num_heads, 1, head_dim
            )
            self.k_cache[layer_idx].scatter_(dim=2, index=indices, src=key_states)
            self.v_cache[layer_idx].scatter_(dim=2, index=indices, src=value_states)
        # Path 2: Prefill (Eager Execution)
        else:
            # Indices grid: [Batch, SeqLen]
            # start_indices [B, 1] + positions [1, SeqLen]
            positions = torch.arange(seq_len, device=key_states.device).view(1, seq_len)
            indices = start_indices.view(bsz, 1) + positions
            indices = indices.view(bsz, 1, seq_len, 1).expand(
                bsz, num_heads, seq_len, head_dim
            )
            self.k_cache[layer_idx].scatter_(dim=2, index=indices, src=key_states)
            self.v_cache[layer_idx].scatter_(dim=2, index=indices, src=value_states)
        # Auto-increment internally for THIS layer only
        # We must construct a vector of shape (bsz,)
        delta = torch.full((bsz,), seq_len, device=key_states.device, dtype=torch.int32)
        self.adjust_seqlens(delta, layer_idx=layer_idx)
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return 0

    def get_mask_sizes(
        self, cache_position: torch.Tensor, layer_idx: int
    ) -> tuple[int, int]:
        return 0, 0

    def get_max_cache_shape(self, layer_idx: int = 0) -> int:
        return 0


class GraphDecoder:
    """
    Manages the lifecycle of CUDA Graph, Static KV Cache, and Sequence Metadata.
    """

    def __init__(
        self,
        batch_size: int,
        max_len: int,
        dtype: torch.dtype,
        device: torch.device,
        # NOTE: Model config
        num_hidden_layers: int,
        num_key_value_heads: int,
        head_dim: int,
        vocab_size: int,
        cache: Optional[StreamingCache] = None,
    ):
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.max_len = max_len
        # StreamingCache
        self.num_layers = num_hidden_layers
        self.num_heads = num_key_value_heads
        self.head_dim = head_dim
        if cache is None:
            self.cache = StreamingCache(
                batch_size=batch_size,
                max_len=max_len,
                dtype=dtype,
                device=device,
                num_hidden_layers=num_hidden_layers,
                num_key_value_heads=num_key_value_heads,
                head_dim=self.head_dim,
            )
        else:
            self.cache = cache
            assert self.cache.batch_size == batch_size, "Cache batch size mismatch."
        # Hard-coded memory addresses for Graph Capture (Fixed Size)
        self.static_input_ids = torch.zeros(
            (batch_size, 1), dtype=torch.long, device=device
        )
        self.static_position_ids = torch.zeros(
            (batch_size, 1), dtype=torch.long, device=device
        )
        self.static_logits = torch.zeros(
            (batch_size, 1, vocab_size), dtype=dtype, device=device
        )
        self.init_graph()

    def init_graph(self):
        """
        Initializes the CUDA Graph for Decoding.
        """
        self.graph = torch.cuda.CUDAGraph()
        self.is_captured = False

    @property
    def cache_seqlens(self) -> torch.Tensor:
        return self.cache.cache_seqlens

    def reset(self):
        """Reset internal cache."""
        self.cache.reset()
        self.static_input_ids.zero_()
        self.static_position_ids.zero_()
        self.static_logits.zero_()

    @torch.inference_mode()
    def capture(self, model: torch.nn.Module):
        """
        Executes Warmup and Captures the Decode Graph.
        """
        if self.is_captured:
            raise RuntimeError("Graph already captured.")

        # 1. Warmup Strategy
        # Set dummy length to avoid kernel errors during capture
        num_warmup = 3
        self.cache_seqlens.fill_(self.max_len - num_warmup - 1)
        model.eval()
        # Run forward pass `num_warmup` times to trigger JIT compilation
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(num_warmup):
                model(
                    input_ids=self.static_input_ids,
                    position_ids=self.static_position_ids,
                    past_key_values=self.cache,
                    attn_cache_seqlens=self.cache_seqlens,
                    use_cache=True,
                )
        torch.cuda.current_stream().wait_stream(s)
        # 2. Graph Capture
        # Only capture the Compute Graph (Forward pass)
        with torch.cuda.graph(self.graph):
            logits = model(
                input_ids=self.static_input_ids,
                position_ids=self.static_position_ids,
                past_key_values=self.cache,
                attn_cache_seqlens=self.cache_seqlens,
                use_cache=True,
            ).logits
            self.static_logits.copy_(logits)
        # Restore state
        self.reset()
        self.is_captured = True

    @torch.inference_mode()
    def step(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Atomic Decode Step:
        1. Copy inputs -> Static Buffer
        2. Replay Graph (which automatically calls cache.update)
        """
        if not self.is_captured:
            raise RuntimeError("Graph not captured.")
        # Strict checking for batch size
        assert input_ids.shape[0] == self.batch_size, (
            f"Input shape {input_ids.shape} mismatch with graph batch size {self.batch_size}"
        )
        # 1. Update Static Inputs (In-place copy full batch)
        self.static_input_ids.copy_(input_ids)
        self.static_position_ids.copy_(position_ids)
        # 2. Replay
        # NOTE: cache_seqlens is auto-incremented inside the model call (cache.update)
        self.graph.replay()
        # Return full logits (No slicing needed)
        return self.static_logits


class StreamingInferenceEngine:
    """
    High-performance wrapper optimized for Online RL Rollouts with Qwen-VL.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        batch_size: int,  # NOTE: This is the strict fixed batch size (bsz * num_gen)
        max_len: int,
        num_hidden_layers: int,
        num_key_value_heads: int,
        head_dim: int,
        vocab_size: int,
        pad_token_id: int,
        eos_token_ids: Union[int, list[int]],
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        self.model = model
        # Special token ids
        self.pad_token_id = pad_token_id
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        self.eos_token_ids = torch.tensor(
            eos_token_ids, device=device, dtype=torch.long
        )
        self.primary_eos_token_id = eos_token_ids[0]
        # Config
        self.device = torch.device(device)
        self.batch_size = batch_size
        # Initialize the Execution Engine
        self.decoder = GraphDecoder(
            batch_size=self.batch_size,
            max_len=max_len,
            dtype=dtype,
            device=device,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            vocab_size=vocab_size,
        )
        # STATE MANAGEMENT: Cache for the NEXT starting position ids.
        # It stores the 'cur_pos_ids' (next token pos) from the END of the previous generate.
        self.next_start_pos: Optional[torch.Tensor] = None
        # Trigger Capture immediately (includes auto-warmup)
        self.decoder.capture(self.model)

    def reset(self):
        """Clears the KV cache and Position ID cache to start a new independent stream."""
        self.decoder.reset()
        self.next_start_pos = None

    def _expand_position_ids(
        self,
        position_ids: torch.Tensor,
        num_generations: int,
    ):
        if num_generations == 1:
            return position_ids
        if position_ids.ndim == 3:
            # Qwen 2.5/3 VL shape: (3, B, L) -> Repeat on dim 1
            return position_ids.repeat_interleave(num_generations, dim=1)
        else:
            # Standard shape: (B, L) -> Repeat on dim 0
            return position_ids.repeat_interleave(num_generations, dim=0)

    def _process_position_ids(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        num_generations: int,
        adjust_position_ids: bool,
    ) -> torch.Tensor:
        """
        Handles expansion and streaming adjustment of position IDs.
        Crucial: position_ids must NOT be None.
        """
        # 1. Expand provided IDs
        position_ids = self._expand_position_ids(position_ids, num_generations)
        # 2. Adjust for History (Streaming)
        if adjust_position_ids and self.next_start_pos is not None:
            if position_ids.ndim == 3:
                offset = self.next_start_pos.unsqueeze(0)
            else:
                offset = self.next_start_pos
            position_ids = position_ids + offset
        return position_ids

    def _expand_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        pixel_values_videos: Optional[torch.Tensor],
        video_grid_thw: Optional[torch.Tensor],
        num_generations: int,
    ):
        """
        Handles the complexity of expanding packed tensors (video features).
        """
        if num_generations == 1:
            return (
                input_ids,
                attention_mask,
                pixel_values_videos,
                video_grid_thw,
            )
        # 1. Expand Standard Tensors (input_ids, attention_mask)
        input_ids = input_ids.repeat_interleave(num_generations, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(num_generations, dim=0)
        # 3. Expand Video Features (Packed Tensor Logic)
        if pixel_values_videos is not None and video_grid_thw is not None:
            # Expand the grid metadata first: (B, 3) -> (B * N, 3)
            video_grid_thw_expanded = video_grid_thw.repeat_interleave(
                num_generations, dim=0
            )
            # Calculate the length of features for each sample in the original batch.
            split_sizes = video_grid_thw.prod(dim=-1).cpu().tolist()
            assert sum(split_sizes) == pixel_values_videos.shape[0]
            # Split packed tensor into list of tensors per sample
            splits = torch.split(pixel_values_videos, split_sizes)
            # Repeat each sample's features
            expanded_splits = [t for t in splits for _ in range(num_generations)]
            # Repack
            pixel_values_videos = torch.cat(expanded_splits, dim=0)
            video_grid_thw = video_grid_thw_expanded
        return (
            input_ids,
            attention_mask,
            pixel_values_videos,
            video_grid_thw,
        )

    def _get_next_position_ids(self, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Derives the starting position ID for the decode phase (scalar per batch item).
        """
        if position_ids.ndim == 3:
            # Case A: Qwen 2.5/3 VL MROPE (3, B, L)
            # Logic: Transpose to (B, 3, L) -> Flatten last dims -> Max -> +1
            return position_ids.transpose(0, 1).flatten(1).max(dim=1)[0] + 1
        else:
            # Case B: Standard (B, L)
            return position_ids.max(dim=1)[0] + 1

    @torch.inference_mode()
    def prefill(
        self,
        *,
        input_ids: torch.Tensor,
        num_generations: int,
        position_ids: torch.Tensor,  # Strict Requirement
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        logits_to_keep: int = 1,
    ) -> torch.Tensor:
        (
            input_ids,
            attention_mask,
            pixel_values_videos,
            video_grid_thw,
        ) = self._expand_inputs(
            input_ids,
            attention_mask,
            pixel_values_videos,
            video_grid_thw,
            num_generations,
        )
        # Verify strict batch size
        assert input_ids.shape[0] == self.batch_size, (
            f"Input batch size {input_ids.shape[0]} does not match strict batch size {self.batch_size}"
        )
        cache_seqlens = self.decoder.cache_seqlens
        # Forward pass (Prefill)
        logits = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=self.decoder.cache,
            use_cache=True,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            logits_to_keep=logits_to_keep,
            attn_cache_seqlens=cache_seqlens,
        ).logits
        # Post-Prefill Correction
        if attention_mask is not None:
            valid_lengths = attention_mask.sum(dim=1).to(
                dtype=torch.int32, device=input_ids.device
            )
            current_seq_len = input_ids.shape[1]
            delta = valid_lengths - current_seq_len
            self.decoder.cache.adjust_seqlens(delta, layer_idx=None)
        return logits

    def sample_one_step(
        self,
        *,
        logits: torch.Tensor,
        repetition_penalty: float,
        temperature: float,
        top_k: int,
        top_p: float,
        generated_ids: Optional[torch.Tensor] = None,
        step: Optional[int] = None,
        generated_tokens: Optional[torch.Tensor] = None,
        generated_length: Optional[torch.Tensor] = None,
        sample: Optional[callable] = None,
        sample_kwargs: Optional[dict] = None,
    ) -> torch.Tensor:
        # Normalize logits shape to [batch_size, vocab_size]
        if logits.ndim == 3:
            logits = logits[:, -1, :]  # [batch_size, vocab_size]
        elif logits.ndim == 2:
            pass
        else:
            raise ValueError(f"Unexpected logits shape: {logits.shape}")

        if repetition_penalty != 1.0 and generated_ids is not None:
            score = torch.gather(logits, 1, generated_ids)
            score = torch.where(
                score < 0, score * repetition_penalty, score / repetition_penalty
            )
            logits.scatter_(1, generated_ids, score)
        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature
        next_token = top_k_top_p_sampling_from_logits(logits, top_k, top_p).unsqueeze(
            -1
        )
        # Apply custom sample function if provided
        if sample is not None:
            sample_kwargs = sample_kwargs or {}
            next_token = sample(
                next_token=next_token,
                logits=logits,
                step=step,
                generated_tokens=generated_tokens,
                generated_length=generated_length,
                **sample_kwargs,
            )
        return next_token

    @torch.inference_mode()
    def generate(
        self,
        *,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,  # Must be provided
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        num_generations: int = 1,
        max_new_tokens: int = 128,
        top_k: int = 50,
        top_p: float = 0.95,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        adjust_position_ids: bool = True,
        sample: Optional[callable] = None,
        sample_kwargs: Optional[dict] = None,
    ) -> List[torch.Tensor]:
        """
        Executes the generation pipeline.
        position_ids: user must provide base position ids for the current input chunk.

        Args:
            sample: Optional custom sampling function that receives:
                - next_token: torch.Tensor [batch_size, 1] - token from standard sampling
                - logits: torch.Tensor [batch_size, vocab_size] - logits used for sampling
                - step: int - current generation step (0-based)
                - generated_tokens: torch.Tensor [batch_size, max_new_tokens+1] - all generated tokens
                - generated_length: torch.Tensor [batch_size] - actual length of generated tokens
                - **sample_kwargs: additional context passed via sample_kwargs dict
            sample_kwargs: Optional dict of additional context to pass to sample function
        """
        # Validate Input Batch Size
        input_bsz = input_ids.shape[0]
        effective_bsz = input_bsz * num_generations
        assert effective_bsz == self.batch_size, (
            f"Total batch size ({effective_bsz}) must strictly match initialized batch size ({self.batch_size})."
        )
        # 1. Process Position IDs (Expand & Shift based on Cache)
        position_ids = self._process_position_ids(
            input_ids=input_ids,
            position_ids=position_ids,
            num_generations=num_generations,
            adjust_position_ids=adjust_position_ids,
        )
        # 2. Prefill
        logits = self.prefill(
            input_ids=input_ids,
            num_generations=num_generations,
            position_ids=position_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )
        # 3. Decode Loop Setup
        full_tokens = torch.full(
            (effective_bsz, max_new_tokens + 1),
            self.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        valid_token_lens = torch.zeros(
            effective_bsz, dtype=torch.long, device=self.device
        )

        next_token = self.sample_one_step(
            logits=logits,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            step=0,
            generated_tokens=full_tokens,
            generated_length=valid_token_lens,
            sample=sample,
            sample_kwargs=sample_kwargs,
        )
        full_tokens[:, 0] = next_token.squeeze(
            -1
        )  # NOTE: Fill the output token from prefill
        valid_token_lens += 1  # Update generated length

        finished = torch.isin(next_token.squeeze(-1), self.eos_token_ids)
        # NOTE: We should ensure that all the eos tokens in the sequences are encoded for next turn generation.
        eos_encoded = torch.zeros(
            (effective_bsz,), dtype=torch.bool, device=self.device
        )
        # Calculate next position ids for decoding.
        # This gives us the scalar START pos for the next token [StrictBatchSize]
        cur_pos_ids = self._get_next_position_ids(position_ids).unsqueeze(1)
        # NOTE: Start from 1, because prefill has one output
        for step in range(1, max_new_tokens + 1):
            logits = self.decoder.step(next_token, cur_pos_ids)
            # NOTE: If step == max_new_tokens, we do not need to get the next token because it
            # is forced to stop.
            if step < max_new_tokens:
                if step == max_new_tokens - 1:
                    # We force set the next token to eos when reaching the max new tokens bound
                    next_token = torch.full(
                        (effective_bsz, 1),
                        self.primary_eos_token_id,
                        dtype=torch.long,
                        device=self.device,
                    )
                else:
                    next_token = self.sample_one_step(
                        logits=logits,
                        repetition_penalty=repetition_penalty,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        step=step,
                        generated_tokens=full_tokens,
                        generated_length=valid_token_lens,
                        sample=sample,
                        sample_kwargs=sample_kwargs,
                    )
                full_tokens[:, step] = torch.where(
                    finished,
                    self.pad_token_id,
                    next_token.squeeze(-1),
                )
                valid_token_lens += ~finished
            # NOTE: If eos is already encoded, then the current step is invalid,
            # so the cache_seqlens and cur_pos_ids should not update.
            self.decoder.cache.adjust_seqlens(
                torch.where(
                    eos_encoded,
                    torch.tensor(-1, dtype=torch.int32, device=self.device),
                    torch.tensor(0, dtype=torch.int32, device=self.device),
                ),
                layer_idx=None,
            )
            cur_pos_ids += (~eos_encoded).long().unsqueeze(1)
            eos_encoded |= finished  # NOTE: We update the eos_encoded
            # We calculate the new finished state
            finished |= torch.isin(next_token.squeeze(-1), self.eos_token_ids)
            if eos_encoded.all():
                break
        assert eos_encoded.all(), (
            "All the sequences should end with <eos> and the <eos> should be stored to kv_cache for correct multi-turn generation."
        )
        # STATE UPDATE: Save the final position id for the next streaming call.
        # cur_pos_ids now holds (last_token_pos + 1).
        self.next_start_pos = cur_pos_ids.detach().clone()
        return [
            full_tokens[i, :length]
            for i, length in enumerate(valid_token_lens.tolist())
        ]


class StreamingWindowInferenceEngine(StreamingInferenceEngine):
    """
    StreamingInferenceEngine with automatic video token sliding window eviction.

    Mirrors the training-time flex_attention sliding window behavior:
    only the most recent `video_flex_window_size` video chunks are kept in the KV cache;
    older video token blocks are evicted before each new chunk is prefilled.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        batch_size: int,
        max_len: int,
        num_hidden_layers: int,
        num_key_value_heads: int,
        head_dim: int,
        vocab_size: int,
        pad_token_id: int,
        eos_token_ids: Union[int, list[int]],
        video_token_id: int,
        video_flex_window_size: int = DEFAULT_VIDEO_FLEX_WINDOW_SIZE,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        super().__init__(
            model=model,
            batch_size=batch_size,
            max_len=max_len,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            eos_token_ids=eos_token_ids,
            dtype=dtype,
            device=device,
        )
        self.video_token_id = video_token_id
        self.video_flex_window_size = video_flex_window_size
        # ---- Tensor-based sliding window bookkeeping (fully vectorized) ----
        # Stores (start, end) cache positions for each video chunk per effective batch item.
        # Shape: [batch_size, video_flex_window_size]
        self._window_starts = torch.zeros(
            (self.batch_size, video_flex_window_size),
            dtype=torch.long,
            device=self.device,
        )
        self._window_ends = torch.zeros(
            (self.batch_size, video_flex_window_size),
            dtype=torch.long,
            device=self.device,
        )
        # Active window count per batch item: [batch_size]
        self._window_count = torch.zeros(
            self.batch_size,
            dtype=torch.long,
            device=self.device,
        )
        # Initialize CacheEviction (shares the same StreamingCache)
        self._cache_eviction = CacheEviction(
            cache=self.decoder.cache,
            batch_size=batch_size,
            max_len=max_len,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            device=torch.device(device),
        )

    def reset(self):
        """Clears KV cache, position cache, and video window bookkeeping."""
        super().reset()
        self._window_starts.zero_()
        self._window_ends.zero_()
        self._window_count.zero_()

    # ------------------------------------------------------------------
    # Internal helpers (fully vectorized, no Python for-loops)
    # ------------------------------------------------------------------
    def _detect_video_tokens(
        self,
        input_ids: torch.Tensor,
        num_generations: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized video token detection.

        Assumptions (guaranteed by caller):
            - Video tokens are contiguous in every sample.
            - Video tokens are never masked (always valid).
            - Every sample contains at least one video token.

        Args:
            input_ids: [input_bsz, seq_len] (before expansion)
            num_generations: int

        Returns:
            num_vt:    [effective_bsz]  number of video tokens per item
            first_pos: [effective_bsz]  index of first video token in input_ids per item
        """
        video_mask = input_ids == self.video_token_id  # [input_bsz, seq_len]
        num_vt = video_mask.sum(dim=1)  # [input_bsz]
        first_pos = video_mask.to(torch.int32).argmax(dim=1)  # [input_bsz]
        if num_generations > 1:
            num_vt = num_vt.repeat_interleave(num_generations)  # [effective_bsz]
            first_pos = first_pos.repeat_interleave(num_generations)
        return num_vt, first_pos

    def _maybe_evict(self):
        """
        Vectorized eviction: for every batch item at capacity, evict the oldest video chunk.
        Items below capacity are no-ops (start == end == 0 → gather identity).
        """
        needs_evict = self._window_count >= self.video_flex_window_size  # [B]
        if not needs_evict.any():
            return

        # Eviction params: oldest window's (start, end) or (0, 0) for no-op items
        evict_starts = torch.where(needs_evict, self._window_starts[:, 0], 0)  # [B]
        evict_ends = torch.where(needs_evict, self._window_ends[:, 0], 0)  # [B]
        evict_lens = (evict_ends - evict_starts).unsqueeze(1)  # [B, 1]

        # 1. Evict from KV cache
        self._cache_eviction.evict(evict_starts, evict_ends)

        # 2. Shift window bookkeeping: roll left by 1 and subtract eviction length
        rolled_starts = torch.roll(self._window_starts, -1, dims=1)  # [B, W]
        rolled_ends = torch.roll(self._window_ends, -1, dims=1)  # [B, W]
        mask = needs_evict.unsqueeze(1)  # [B, 1]  broadcast over W
        self._window_starts = torch.where(
            mask, rolled_starts - evict_lens, self._window_starts
        )
        self._window_ends = torch.where(
            mask, rolled_ends - evict_lens, self._window_ends
        )
        self._window_count -= needs_evict.long()

    def _record_video_windows(
        self,
        cache_lens_before: torch.Tensor,
        num_vt: torch.Tensor,
        first_pos: torch.Tensor,
    ):
        """
        Vectorized recording of new video token windows into the bookkeeping tensors.

        Args:
            cache_lens_before: [effective_bsz]  cache length snapshot before prefill
            num_vt:            [effective_bsz]  number of video tokens per item
            first_pos:         [effective_bsz]  first video token position in input_ids
        """
        new_start = cache_lens_before + first_pos  # [B]
        new_end = new_start + num_vt  # [B]
        idx = self._window_count.unsqueeze(1)  # [B, 1]
        self._window_starts.scatter_(1, idx, new_start.unsqueeze(1))
        self._window_ends.scatter_(1, idx, new_end.unsqueeze(1))
        self._window_count += 1

    # ------------------------------------------------------------------
    # Public API  (override)
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate(
        self,
        *,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        num_generations: int = 1,
        max_new_tokens: int = 128,
        top_k: int = 50,
        top_p: float = 0.95,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        adjust_position_ids: bool = True,
        sample: Optional[callable] = None,
        sample_kwargs: Optional[dict] = None,
    ) -> List[torch.Tensor]:
        effective_bsz = input_ids.shape[0] * num_generations

        # 1. Detect video tokens (vectorized, no loops)
        num_vt, first_pos = self._detect_video_tokens(input_ids, num_generations)

        # 2. Evict oldest video block for any batch item at capacity (vectorized)
        self._maybe_evict()

        # 3. Snapshot per-item cache lengths *after* eviction, *before* prefill
        cache_lens_before = self.decoder.cache_seqlens[0, :effective_bsz].clone()  # [B]

        # 4. Delegate to parent generate
        outputs = super().generate(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            num_generations=num_generations,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            adjust_position_ids=adjust_position_ids,
            sample=sample,
            sample_kwargs=sample_kwargs,
        )

        # 5. Record the new video token windows (vectorized).
        # Guard: skip when the input contained no video tokens to avoid
        # ghost zero-length window entries that would corrupt bookkeeping.
        if num_vt.any():
            self._record_video_windows(cache_lens_before, num_vt, first_pos)

        return outputs


class CacheEviction:
    """
    CUDA Graph-optimized KV Cache Eviction.
    Prunes the static KV cache by moving cache entries from [end:] to [start:],
    effectively compacting the cache to reduce memory usage.
    """

    def __init__(
        self,
        cache: StreamingCache,
        batch_size: int,
        max_len: int,
        num_hidden_layers: int,
        num_key_value_heads: int,
        head_dim: int,
        device: torch.device,
    ):
        self.cache = cache
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_key_value_heads
        self.head_dim = head_dim
        self.device = device

        # Pre-compute Gather Template [1, 1, MaxLen, 1]
        self.pos_template = torch.arange(
            self.max_len, device=self.device, dtype=torch.long
        ).view(1, 1, -1, 1)

        # Static buffers for graph capture
        # start and end indices: [num_hidden_layers, batch_size]
        self.static_starts = torch.zeros(
            (num_hidden_layers, batch_size), dtype=torch.long, device=device
        )
        self.static_ends = torch.zeros(
            (num_hidden_layers, batch_size), dtype=torch.long, device=device
        )

        # Initialize graph
        self.graph = torch.cuda.CUDAGraph()
        self.is_captured = False
        self.capture()

    def _evict_kernel(self, starts: torch.Tensor, ends: torch.Tensor):
        """
        In-place eviction kernel for [B, H, L, D] cache tensors.

        Args:
            starts: [num_hidden_layers, batch_size] - target start indices
            ends: [num_hidden_layers, batch_size] - source end indices
        """
        num_layers, bsz = starts.shape

        for layer_idx in range(num_layers):
            k_cache = self.cache.k_cache[layer_idx]  # [B, H, L, D]
            v_cache = self.cache.v_cache[layer_idx]  # [B, H, L, D]
            current_len = self.cache.cache_seqlens[layer_idx]  # [B]

            layer_starts = starts[layer_idx]  # [B]
            layer_ends = ends[layer_idx]  # [B]

            # Calculate tokens to keep
            tokens_to_keep = torch.clamp(current_len - layer_ends, min=0)  # [B]
            new_lengths = layer_starts + tokens_to_keep  # [B]

            # Compute gather indices (vectorized)
            starts_exp = layer_starts.view(bsz, 1, 1, 1)  # [B, 1, 1, 1]
            ends_exp = layer_ends.view(bsz, 1, 1, 1)  # [B, 1, 1, 1]
            shift = ends_exp - starts_exp  # [B, 1, 1, 1]

            # Mask positions that need to be moved
            mask = self.pos_template >= starts_exp  # [B, 1, L, 1]
            tokens_to_keep_exp = tokens_to_keep.view(bsz, 1, 1, 1)  # [B, 1, 1, 1]
            move_upper_bound = starts_exp + tokens_to_keep_exp  # [B, 1, 1, 1]
            move_mask = mask & (self.pos_template < move_upper_bound)  # [B, 1, L, 1]

            # Compute gather indices: move from end+(pos-start) if needed, else keep original
            gather_indices = self.pos_template + (move_mask * shift)  # [B, 1, L, 1]
            gather_indices = gather_indices.clamp(min=0, max=self.max_len - 1)

            # Expand to match cache shape
            B, H, L, D = k_cache.shape
            final_indices = gather_indices.expand(B, H, L, D)  # [B, H, L, D]

            # Gather and copy in-place
            shifted_k = torch.gather(k_cache, dim=2, index=final_indices)
            shifted_v = torch.gather(v_cache, dim=2, index=final_indices)
            k_cache.copy_(shifted_k)
            v_cache.copy_(shifted_v)

            # Zero out positions beyond new_lengths
            new_lengths_exp = new_lengths.view(bsz, 1, 1, 1)  # [B, 1, 1, 1]
            zero_mask = self.pos_template >= new_lengths_exp  # [B, 1, L, 1]
            zero_mask_expanded = zero_mask.expand(B, H, L, D)  # [B, H, L, D]
            k_cache[zero_mask_expanded] = 0.0
            v_cache[zero_mask_expanded] = 0.0

            # Update cache_seqlens in-place
            self.cache.cache_seqlens[layer_idx].copy_(new_lengths)

    @torch.inference_mode()
    def capture(self):
        """
        Captures the CUDA graph for eviction operation.
        """
        if self.is_captured:
            raise RuntimeError("Graph already captured.")

        print("Capturing CUDA Graph for Cache Eviction...")
        # Warmup
        num_warmup = 3
        for _ in range(num_warmup):
            # Set dummy values for warmup
            self.static_starts.fill_(100)
            self.static_ends.fill_(164)
            # Set cache to some dummy state
            for layer_idx in range(self.num_hidden_layers):
                self.cache.cache_seqlens[layer_idx].fill_(200)
            self._evict_kernel(self.static_starts, self.static_ends)

        # Capture graph
        with torch.cuda.graph(self.graph):
            self._evict_kernel(self.static_starts, self.static_ends)

        print("Capture Done.")
        self.is_captured = True
        # Reset cache and static buffers after capture
        self.cache.reset()
        self.static_starts.zero_()
        self.static_ends.zero_()

    @torch.inference_mode()
    def evict(self, start: torch.Tensor, end: torch.Tensor):
        """
        Evicts KV cache entries using CUDA graph.

        Args:
            start: [batch_size] or [num_hidden_layers, batch_size] - target start indices
            end: [batch_size] or [num_hidden_layers, batch_size] - source end indices

        This operation:
        1. Moves cache entries from [end:] to [start:] for each batch item
        2. Updates cache_seqlens in-place
        3. Does NOT modify position encodings
        """
        if not self.is_captured:
            raise RuntimeError("Graph not captured.")

        # Normalize inputs to [num_hidden_layers, batch_size]
        if start.ndim == 1:
            start = start.unsqueeze(0).expand(self.num_hidden_layers, self.batch_size)
        if end.ndim == 1:
            end = end.unsqueeze(0).expand(self.num_hidden_layers, self.batch_size)

        # Validate inputs
        assert start.shape == (
            self.num_hidden_layers,
            self.batch_size,
        ), f"start shape {start.shape} != ({self.num_hidden_layers}, {self.batch_size})"
        assert end.shape == (
            self.num_hidden_layers,
            self.batch_size,
        ), f"end shape {end.shape} != ({self.num_hidden_layers}, {self.batch_size})"
        assert (start >= 0).all(), "start indices must be >= 0"
        assert (end >= 0).all(), "end indices must be >= 0"
        assert (start <= self.max_len).all(), "start indices must be <= max_len"
        assert (end <= self.max_len).all(), "end indices must be <= max_len"

        # Copy inputs to static buffers
        self.static_starts.copy_(start)
        self.static_ends.copy_(end)

        # Replay graph
        self.graph.replay()


def think_budget_sample(
    next_token: torch.Tensor,
    logits: torch.Tensor,
    step: int,
    generated_tokens: torch.Tensor,
    generated_length: torch.Tensor,
    think_end_token_id: int,
    max_think_tokens: int,
    **kwargs,
) -> torch.Tensor:
    """
    Custom sample function that enforces a thinking-token budget.

    During generation, for each sequence in the batch this function checks:
      1. Whether ``</think>`` (identified by *think_end_token_id*) has already
         been produced in the tokens generated so far.
      2. Whether the number of tokens generated so far has reached the budget
         *max_think_tokens*.

    The budget counts **all** generated tokens in this phase, including
    ``<think>``, the thinking content, and ``</think>`` (i.e. *generated_length*
    is the total number of tokens produced so far).

    If a sequence exceeds the budget **and** has not yet emitted ``</think>``,
    the sampled token is forcibly replaced with ``</think>`` so that the model
    can transition to producing the final answer within the remaining output
    space.

    This function conforms to the ``sample`` callback API of
    :meth:`StreamingInferenceEngine.generate` and is intended to be passed via
    the ``sample`` / ``sample_kwargs`` arguments::

        engine.generate(
            ...,
            sample=think_budget_sample,
            sample_kwargs={
                "think_end_token_id": tokenizer.convert_tokens_to_ids("</think>"),
                "max_think_tokens": 512,
            },
        )

    Args:
        next_token:          [batch_size, 1]  – token produced by standard sampling.
        logits:              [batch_size, vocab_size] – logits (for reference; unused here).
        step:                Current generation step (0-based).
        generated_tokens:    [batch_size, max_new_tokens+1] – all tokens generated so far
                             (indices ``0 .. step-1`` are valid).
        generated_length:    [batch_size] – actual number of valid generated tokens per sequence.
        think_end_token_id:  Token ID that represents ``</think>``.
        max_think_tokens:    Maximum number of tokens a sequence may generate before
                             ``</think>`` is forced.

    Returns:
        [batch_size, 1] – Possibly modified *next_token* tensor.
    """
    # 1. Check which sequences have already emitted </think>
    #    generated_tokens[:, :step] holds tokens from previous steps;
    #    the token for the current step has NOT been written yet.
    if step > 0:
        has_think_end = (generated_tokens[:, :step] == think_end_token_id).any(
            dim=1
        )  # [B]
    else:
        has_think_end = torch.zeros(
            next_token.shape[0], dtype=torch.bool, device=next_token.device
        )

    # 2. Check which sequences have exhausted the thinking budget
    over_budget = generated_length >= max_think_tokens  # [B]

    # 3. Force </think> for sequences that are over budget AND haven't emitted it
    force_mask = over_budget & ~has_think_end  # [B]

    if force_mask.any():
        think_end = torch.full_like(next_token, think_end_token_id)  # [B, 1]
        next_token = torch.where(force_mask.unsqueeze(1), think_end, next_token)

    return next_token


def think_budget_sample_restricted(
    next_token: torch.Tensor,
    logits: torch.Tensor,
    step: int,
    generated_tokens: torch.Tensor,
    generated_length: torch.Tensor,
    think_end_token_id: int,
    max_think_tokens: int,
    eos_token_id: int,
    silent_token_id: int,
    response_token_id: int,
    restricted_token_ids: list[int],
    allow_deferral: bool = False,
    is_query_window: bool = True,
    **kwargs,
) -> torch.Tensor:
    """
    Extended version of :func:`think_budget_sample` that additionally enforces
    a deterministic output pattern **after** ``</think>`` is emitted.

    The function first delegates to :func:`think_budget_sample` to enforce the
    thinking-token budget (forcing ``</think>`` when over budget).  Then, once
    ``</think>`` has appeared in the generated history, it forces a fixed
    sequence of tokens depending on whether *restricted_token_ids* is provided:

    **When ``restricted_token_ids is None``** (unrestricted / silent mode)::

        </think>  →  <silent>  →  <|im_end|>

    **When ``restricted_token_ids is not None``** (constrained vocabulary)::

        </think>  →  <response>  →  argmax(logits[restricted_token_ids])  →  <|im_end|>

    Usage::

        # Silent mode
        engine.generate(
            ...,
            sample=think_budget_sample_restricted,
            sample_kwargs={
                "think_end_token_id": tokenizer.convert_tokens_to_ids("</think>"),
                "max_think_tokens": 512,
                "eos_token_id": tokenizer.convert_tokens_to_ids("<|im_end|>"),
                "silent_token_id": tokenizer.convert_tokens_to_ids("<silent>"),
                "response_token_id": tokenizer.convert_tokens_to_ids("<response>"),
                # restricted_token_ids defaults to None → silent mode
            },
        )

        # Constrained vocabulary mode
        engine.generate(
            ...,
            sample=think_budget_sample_restricted,
            sample_kwargs={
                "think_end_token_id": tokenizer.convert_tokens_to_ids("</think>"),
                "max_think_tokens": 512,
                "eos_token_id": tokenizer.convert_tokens_to_ids("<|im_end|>"),
                "silent_token_id": tokenizer.convert_tokens_to_ids("<silent>"),
                "response_token_id": tokenizer.convert_tokens_to_ids("<response>"),
                "restricted_token_ids": [id_A, id_B, id_C, ...],
            },
        )

    Args:
        next_token:           [batch_size, 1]  – token from standard sampling.
        logits:               [batch_size, vocab_size] – logits for the current step.
        step:                 Current generation step (0-based).
        generated_tokens:     [batch_size, max_new_tokens+1] – tokens generated so far
                              (indices ``0 .. step-1`` are valid).
        generated_length:     [batch_size] – number of valid generated tokens per sequence.
        think_end_token_id:   Token ID for ``</think>``.
        max_think_tokens:     Budget: max tokens before ``</think>`` is forced.
        eos_token_id:         Token ID for ``<|im_end|>``.
        silent_token_id:      Token ID for ``<silent>``.
        response_token_id:    Token ID for ``<response>``.
        restricted_token_ids: Optional list of allowed answer token IDs.  When *None*,
                              the silent-mode pattern is used; otherwise the constrained
                              vocabulary pattern is used.
        allow_deferral:       If True, allows the model to choose between answering
                              and deferring (silent) when in a query window.
        is_query_window:      If False, forces silent mode (observation).

    Returns:
        [batch_size, 1] – Possibly modified *next_token* tensor.
    """
    # ---- Phase 1: enforce thinking budget via think_budget_sample ----
    next_token = think_budget_sample(
        next_token=next_token,
        logits=logits,
        step=step,
        generated_tokens=generated_tokens,
        generated_length=generated_length,
        think_end_token_id=think_end_token_id,
        max_think_tokens=max_think_tokens,
    )

    # ---- Phase 2: enforce post-</think> output pattern ----
    # We need at least one token in history to detect </think>.
    if step == 0:
        return next_token

    device = next_token.device
    history = generated_tokens[:, :step]  # [B, step]

    # Locate the FIRST </think> in each sequence's history.
    think_mask = history == think_end_token_id  # [B, step]
    has_think = think_mask.any(dim=1)  # [B]

    if not has_think.any():
        return next_token

    # Position of the first </think>; argmax on a bool tensor returns the
    # index of the first True.  For sequences without </think> the value is
    # meaningless but guarded by has_think below.
    think_pos = think_mask.to(torch.long).argmax(dim=1)  # [B]

    # Number of tokens that have been WRITTEN after </think>.
    # Example: if </think> is at index 3 and step==4, one token (index 3) is
    # </think> itself, and tokens_after == step - think_pos - 1 == 0, meaning
    # nothing has been written after it yet, so the CURRENT next_token is the
    # first token after </think>.
    tokens_after = (step - think_pos - 1) * has_think.long()  # [B]

    if not is_query_window:
        # ---- Silent mode: </think> → <silent> → <|im_end|> ----
        # tokens_after == 0  →  force <silent>
        # tokens_after == 1  →  force <|im_end|>
        force_silent = has_think & (tokens_after == 0)  # [B]
        force_eos = has_think & (tokens_after == 1)  # [B]

        if force_silent.any():
            silent_t = torch.full_like(next_token, silent_token_id)
            next_token = torch.where(force_silent.unsqueeze(1), silent_t, next_token)
        if force_eos.any():
            eos_t = torch.full_like(next_token, eos_token_id)
            next_token = torch.where(force_eos.unsqueeze(1), eos_t, next_token)
    elif not allow_deferral:
        # ---- Constrained mode: </think> → <response> → top-1 restricted → <|im_end|> ----
        # tokens_after == 0  →  force <response>
        # tokens_after == 1  →  force argmax over restricted_token_ids
        # tokens_after == 2  →  force <|im_end|>
        force_response = has_think & (tokens_after == 0)  # [B]
        force_restricted = has_think & (tokens_after == 1)  # [B]
        force_eos = has_think & (tokens_after == 2)  # [B]

        if force_response.any():
            response_t = torch.full_like(next_token, response_token_id)
            next_token = torch.where(
                force_response.unsqueeze(1), response_t, next_token
            )
        if force_restricted.any():
            # Pick the highest-probability token from the restricted vocabulary.
            restricted_ids = torch.tensor(
                restricted_token_ids, device=device, dtype=torch.long
            )
            restricted_logits = logits[:, restricted_ids]  # [B, R]
            top1_local = restricted_logits.argmax(dim=-1)  # [B]
            top1_token = restricted_ids[top1_local]  # [B]
            next_token = torch.where(
                force_restricted.unsqueeze(1), top1_token.unsqueeze(1), next_token
            )
        if force_eos.any():
            eos_t = torch.full_like(next_token, eos_token_id)
            next_token = torch.where(force_eos.unsqueeze(1), eos_t, next_token)
    else:
        # ---- Deferral mode: </think> → (model chooses) ----
        # If model chooses <response>: → top-1 restricted → <|im_end|>
        # If model chooses <silent>: → <|im_end|>

        # tokens_after == 0: Do NOT force anything. Let model sample.

        # Check what was chosen at (think_pos + 1) for those who are past it.
        # This is only valid if tokens_after >= 1.
        is_past_first = has_think & (tokens_after >= 1)

        if is_past_first.any():
            # Get the first token after </think> from history
            batch_indices = torch.arange(next_token.size(0), device=device)
            # We want generated_tokens[b, think_pos[b] + 1]
            first_token_after = generated_tokens[batch_indices, think_pos + 1]

            # Path A: <response> chosen
            path_response = is_past_first & (first_token_after == response_token_id)
            force_restricted = path_response & (tokens_after == 1)
            force_eos_A = path_response & (tokens_after == 2)

            # Path B: <silent> chosen
            path_silent = is_past_first & (first_token_after == silent_token_id)
            force_eos_B = path_silent & (tokens_after == 1)

            if force_restricted.any():
                restricted_ids = torch.tensor(
                    restricted_token_ids, device=device, dtype=torch.long
                )
                restricted_logits = logits[:, restricted_ids]  # [B, R]
                top1_local = restricted_logits.argmax(dim=-1)  # [B]
                top1_token = restricted_ids[top1_local]  # [B]
                next_token = torch.where(
                    force_restricted.unsqueeze(1), top1_token.unsqueeze(1), next_token
                )

            if force_eos_A.any():
                eos_t = torch.full_like(next_token, eos_token_id)
                next_token = torch.where(force_eos_A.unsqueeze(1), eos_t, next_token)

            if force_eos_B.any():
                eos_t = torch.full_like(next_token, eos_token_id)
                next_token = torch.where(force_eos_B.unsqueeze(1), eos_t, next_token)

    return next_token


@torch.inference_mode()
def streaming_video_chat(
    engine: "StreamingWindowInferenceEngine",
    processor: Any,
    video_path: str,
    *,
    queries: Optional[List[dict]] = None,
    video_start: float = 0.0,
    video_end: Optional[float] = None,
    frames_per_chunk: int = _FRAMES_PER_CHUNK,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
    min_pixels: int = DEFAULT_INFERENCE_MIN_PIXELS,
    max_pixels: int = DEFAULT_INFERENCE_MAX_PIXELS,
    max_new_tokens: int = 128,
    top_k: int = 50,
    top_p: float = 0.95,
    temperature: float = 1.0,
    repetition_penalty: float = 1.0,
    num_generations: int = 1,
    system_prompt: str = "",
    chat_template_wo_system: Optional[str] = None,
    sample: Optional[callable] = None,
    sample_kwargs: Optional[dict] = None,
    break_on_answer: bool = True,
    reset_engine: bool = True,
    extra_processor_keys_to_remove: tuple = ("second_per_grid_ts",),
    model_type: str,
    preloaded_video: Optional[dict] = None,
    slack_time: float = 3.0,
):
    """
    High-level streaming video inference **generator**.

    Reads a video, splits it into temporal chunks, feeds each chunk through
    a :class:`StreamingWindowInferenceEngine`, and **yields** a result dict
    after every chunk.  Queries (user questions / prompts) are injected into
    the chunk whose time window contains the query's timestamp.

    **Workflow per chunk**:

    1. Extract video frames for the current time window.
    2. Build a chat-style text prompt (system prompt on the first chunk only;
       query text appended to whichever chunk's window covers the query
       timestamp).
    3. Tokenise via *processor* and compute position IDs via ``compute_position_ids``.
    4. Call ``engine.generate`` with the appropriate sample callback / kwargs.
    5. **Yield** the result dict to the caller.

    **Operating logic**:
    - **break_on_answer=True** (default, e.g. for Inference/Eval):
      Iterates chunks. If a query is active, it decides whether to force an answer
      based on ``slack_time`` and deadlines.
      - **Observation**: Before query or after query but model chose silent (deferred).
      - **Answer**: When deadline hit or model chose to answer.
      Stops after the first answer.
      Requires ``num_generations=1``.

    - **break_on_answer=False** (e.g. for Training/Rollout):
      Iterates all chunks. Does NOT apply slack time logic.
      Treats every chunk as a generation opportunity (free generation).
      Allows ``num_generations > 1``.
      ``is_answer`` in the result is always ``None``.

    Args:
        engine:
            A pre-initialised :class:`StreamingWindowInferenceEngine`.
        processor:
            HuggingFace processor.
        video_path:
            Path to video file.
        queries:
            List of user queries ({"content": str, "timestamp": float}).
        video_start / video_end:
            Video range.
        frames_per_chunk / max_chunks:
            Chunking config.
        min_pixels / max_pixels:
            Resolution limits.
        max_new_tokens:
            Maximum new tokens for generation.
        top_k / top_p / temperature / repetition_penalty:
            Sampling parameters.
        num_generations:
            Number of parallel generations per input.
            Must be 1 if ``break_on_answer=True``.
        system_prompt:
            System message for first chunk.
        chat_template_wo_system:
            Template for subsequent chunks.
        sample / sample_kwargs:
            Sampling function and kwargs.
            If ``break_on_answer=True``, ``is_query_window`` and ``allow_deferral``
            will be injected into ``sample_kwargs`` to control behavior.
        break_on_answer:
            If ``True``, stop after first answer.
        reset_engine:
            Reset engine before start.
        extra_processor_keys_to_remove:
            Keys to remove from inputs.
        model_type:
            Model type string.
        preloaded_video:
            Preloaded video metadata.
        slack_time:
            Time window to allow deferred answering (only if ``break_on_answer=True``).

    Yields:
        Dict per chunk:
        {
            "chunk_idx": int,
            "is_answer": bool | None,  # True/False if break_on_answer=True, else None
            "has_query": bool,
            "generated_tokens": List[torch.Tensor],
            "window_start": float,
            "window_end": float,
        }
    """

    if reset_engine:
        engine.reset()

    device = engine.device

    # ------------------------------------------------------------------ #
    # 0. Normalise queries                                                #
    # ------------------------------------------------------------------ #
    if queries is None:
        queries = []
    pending_queries: List[dict] = sorted(queries, key=lambda q: q["timestamp"])

    # ------------------------------------------------------------------ #
    # 1-4. Metadata, chunking, and frame loading                          #
    # ------------------------------------------------------------------ #
    if preloaded_video is not None:
        video_end = preloaded_video["video_end"]
        video_start = preloaded_video["video_start"]
        video_chunk_size = preloaded_video["video_chunk_size"]
        num_iterations = preloaded_video["num_iterations"]
        frames_per_chunk = preloaded_video["frames_per_chunk"]
        split_videos = preloaded_video["split_videos"]
        video_kwargs = preloaded_video["video_kwargs"]
        chunk_metadatas = preloaded_video["chunk_metadatas"]
    else:
        _pv = preload_video(
            video_path,
            video_start=video_start,
            video_end=video_end,
            frames_per_chunk=frames_per_chunk,
            max_chunks=max_chunks,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            processor=processor,
            model_type=model_type,
        )
        video_end = _pv["video_end"]
        video_chunk_size = _pv["video_chunk_size"]
        num_iterations = _pv["num_iterations"]
        frames_per_chunk = _pv["frames_per_chunk"]
        split_videos = _pv["split_videos"]
        video_kwargs = _pv["video_kwargs"]
        chunk_metadatas = _pv["chunk_metadatas"]

    is_qwen3vl = model_type == "qwen3vl"

    # Extract single-chunk fps for per-iteration processor calls.
    # load_video_frames replicates fps to len(split_videos), so [0] is safe.
    _per_chunk_vkw: dict = {}
    for _k, _v in video_kwargs.items():
        if isinstance(_v, list) and len(_v) >= 1:
            _per_chunk_vkw[_k] = [_v[0]]
        else:
            _per_chunk_vkw[_k] = _v

    # ------------------------------------------------------------------ #
    # 5. Streaming loop                                                   #
    # ------------------------------------------------------------------ #
    active_query_ts = None

    for iter_idx in range(num_iterations):
        # --- Time window ---
        w_start = video_start + iter_idx * video_chunk_size
        w_end = video_start + (iter_idx + 1) * video_chunk_size
        if iter_idx == num_iterations - 1:
            w_end = video_end

        # --- Per-chunk frames from pre-loaded split ---
        current_frames = split_videos[iter_idx]

        # --- Collect queries whose timestamp falls within [w_start, w_end) ---
        # For the LAST iteration, we also capture queries at exactly w_end
        # (i.e. queries whose timestamp >= w_start and <= w_end).
        is_last_iter = iter_idx == num_iterations - 1
        chunk_queries: List[str] = []
        while pending_queries:
            ts = pending_queries[0]["timestamp"]
            if ts < w_start:
                # Query before current window – consume and skip
                pending_queries.pop(0)
                continue
            if ts < w_end:
                active_query_ts = ts
                chunk_queries.append(pending_queries.pop(0)["content"])
            else:
                break

        has_query = len(chunk_queries) > 0

        # --- Message construction ---
        video_content: dict = {
            "type": "video",
            "video": video_path,
            "video_start": w_start,
            "video_end": w_end,
            "nframes": frames_per_chunk,
            "min_pixels": min_pixels,
            "max_pixels": max_pixels,
        }

        is_first = iter_idx == 0
        user_content: list = [video_content]

        if chunk_queries:
            user_content.append(
                {"type": "text", "text": "\n" + "\n".join(chunk_queries)}
            )

        if is_first:
            messages: list = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_content})
            text_prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            messages = [{"role": "user", "content": user_content}]
            template_kwargs: dict = {}
            if chat_template_wo_system is not None:
                template_kwargs["chat_template"] = chat_template_wo_system
            text_prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )

        # --- Tokenise & compute position IDs ---
        proc_kwargs = dict(
            text=[text_prompt],
            videos=[current_frames],
            return_tensors="pt",
            **_per_chunk_vkw,
        )
        if is_qwen3vl:
            proc_kwargs["do_resize"] = False
            if chunk_metadatas is not None:
                proc_kwargs["video_metadata"] = [chunk_metadatas[iter_idx]]
        inputs = processor(**proc_kwargs)

        inputs_for_rope = dict(inputs)
        inputs_for_rope["video_chunk_size"] = video_chunk_size
        inputs["position_ids"] = compute_position_ids(
            inputs_for_rope,
            processor,
            model_type,
        )

        for key in extra_processor_keys_to_remove:
            inputs.pop(key, None)

        inputs = inputs.to(device)

        # --- Select sample config ---
        cur_sample_kwargs = sample_kwargs.copy() if sample_kwargs is not None else {}
        is_query_window = None
        allow_deferral = None

        if break_on_answer:
            if num_generations != 1:
                raise ValueError(
                    "When break_on_answer=True, num_generations must be 1."
                )

            if active_query_ts is not None:
                is_query_window = True

                time_since_query = w_end - active_query_ts
                if time_since_query <= slack_time and not is_last_iter:
                    # Still in slack window: allow deferral
                    allow_deferral = True
                else:
                    # Deadline passed or last iter: force answer
                    allow_deferral = False
            else:
                # Observation phase (no active query yet)
                is_query_window = False
                allow_deferral = True

        # Pass flags to sample function
        cur_sample_kwargs["is_query_window"] = is_query_window
        cur_sample_kwargs["allow_deferral"] = allow_deferral

        # --- Generate ---
        generated = engine.generate(
            **inputs,
            num_generations=num_generations,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            sample=sample,
            sample_kwargs=cur_sample_kwargs,
        )

        # Determine yielded is_answer
        is_answer_yield = None
        if break_on_answer:
            resp_id = cur_sample_kwargs.get("response_token_id")
            if resp_id is not None:
                gen_tensor = generated[0]
                # Check if response token is present
                is_answer_yield = (gen_tensor == resp_id).any().item()
            else:
                # If we don't have the token ID, we can't determine it reliably this way.
                # But let's stick to the requested logic.
                raise ValueError(
                    "response_token_id missing in sample_kwargs for break_on_answer=True"
                )

            if not allow_deferral and not is_answer_yield:
                raise RuntimeError(
                    "Model failed to answer when deferral was disallowed (Deadline exceeded)."
                )

        yield {
            "chunk_idx": iter_idx,
            "is_answer": is_answer_yield,
            "has_query": has_query,
            "generated_tokens": generated,
            "window_start": w_start,
            "window_end": w_end,
        }

        if break_on_answer and is_answer_yield:
            break
