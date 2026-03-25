import torch
from slyme.utils.pytree import P
from slyme.context import Ref
from deepslyme.context.metadata import ARG, Arg
from deepslyme.node.distributed import DistributedState
from thinkstream.data.stream_data_processor import (
    DEFAULT_MAX_CHUNKS,
    DEFAULT_INFERENCE_MIN_PIXELS,
    DEFAULT_INFERENCE_MAX_PIXELS,
)


def default_scope():
    return {
        "seed": Ref[int]("args.train.seed", metadata={ARG: Arg(default=42)}),
        "deepspeed_path": Ref[str](
            "args.train.deepspeed", metadata={ARG: Arg(required=True)}
        ),
        "process_index": Ref[int]("distributed.state", key_path=tuple(P.process_index)),
        "distributed_state": Ref[DistributedState]("distributed.state"),
        "device": Ref("distributed.state", key_path=tuple(P.device)),
        "deepspeed_config": Ref[dict]("distributed.deepspeed_config"),
        "num_processes": Ref("distributed.state", key_path=tuple(P.num_processes)),
        "ddp_timeout": Ref[int](
            "args.train.ddp_timeout", metadata={ARG: Arg(default=1800)}
        ),
        "bf16": Ref[bool]("args.train.bf16", metadata={ARG: Arg(default=True)}),
        "fp16": Ref[bool]("args.train.fp16", metadata={ARG: Arg(default=False)}),
        "mixed_precision_dtype": Ref[str](
            "args.train.mixed_precision_dtype", metadata={ARG: Arg(default="bf16")}
        ),
        "model": Ref("model"),
        "processor": Ref("processor"),
        "tokenizer": Ref("processor", key_path=tuple(P.tokenizer)),
        "model_for_training": Ref("model_for_training"),
        "data_collator": Ref("data_collator"),
        "num_workers": Ref[int](
            "args.train.dataloader.num_workers", metadata={ARG: Arg(default=0)}
        ),
        "pin_memory": Ref[bool](
            "args.train.dataloader.pin_memory", metadata={ARG: Arg(default=True)}
        ),
        "persistent_workers": Ref[bool](
            "args.train.dataloader.persistent_workers",
            metadata={ARG: Arg(default=False)},
        ),
        "drop_last": Ref[bool](
            "args.train.dataloader.drop_last", metadata={ARG: Arg(default=False)}
        ),
        "prefetch_factor": Ref[int](
            "args.train.dataloader.prefetch_factor",
            metadata={ARG: Arg(default=None)},
        ),
        "arg_max_steps": Ref[int](
            "args.train.max_steps", metadata={ARG: Arg(default=-1)}
        ),
        "arg_num_train_epochs": Ref[float](
            "args.train.num_train_epochs",
            metadata={ARG: Arg(required=True, type=float)},
        ),
        "grad_acc_steps": Ref[int](
            "args.train.gradient_accumulation_steps", metadata={ARG: Arg(default=1)}
        ),
        "output_dir": Ref[str](
            "args.train.output_dir", metadata={ARG: Arg(required=True)}
        ),
        "state_num_train_epochs": Ref[int]("state.train.num_train_epochs"),
        "state_max_steps": Ref[int]("state.train.max_steps"),
        "state_log_history": Ref[list]("state.train.log_history"),
        "state_global_step": Ref[int]("state.train.global_step"),
        "state_epoch_idx": Ref[int]("state.train.epoch_idx"),
        "state_epoch": Ref[float]("state.train.epoch"),
        "control_should_stop_training": Ref[bool]("control.stop_training"),
        "control_should_stop_epoch": Ref[bool]("control.stop_epoch"),
        "save_steps": Ref[int](
            "args.train.save_steps", metadata={ARG: Arg(default=1000)}
        ),
        "weight_decay": Ref[float](
            "args.train.weight_decay", metadata={ARG: Arg(default=0.0)}
        ),
        "optimizer_cls": Ref[str](
            "optimizer_cls", metadata={ARG: Arg(default="adamw")}
        ),
        "optimizer_kwargs": Ref[dict]("optimizer_kwargs"),
        "optimizer": Ref("optimizer"),
        "learning_rate": Ref[float](
            "args.train.learning_rate", metadata={ARG: Arg(default=2e-7)}
        ),
        "adam_beta1": Ref[float](
            "args.train.adam_beta1", metadata={ARG: Arg(default=0.9)}
        ),
        "adam_beta2": Ref[float](
            "args.train.adam_beta2", metadata={ARG: Arg(default=0.999)}
        ),
        "adam_epsilon": Ref[float](
            "args.train.adam_epsilon", metadata={ARG: Arg(default=1e-8)}
        ),
        "lr_scheduler_type": Ref[str](
            "args.train.lr_scheduler_type", metadata={ARG: Arg(default="cosine")}
        ),
        "lr_scheduler_kwargs": Ref[dict](
            "args.train.lr_scheduler_kwargs",
            metadata={ARG: Arg(default_factory=lambda: {})},
        ),
        "lr_scheduler": Ref("lr_scheduler"),
        "warmup_ratio": Ref[float](
            "args.train.warmup_ratio", metadata={ARG: Arg(default=0.0)}
        ),
        "grad_ckpt_kwargs": Ref[dict](
            "args.train.gradient_checkpointing_kwargs",
            metadata={ARG: Arg(default=None)},
        ),
        "progress": Ref("progress"),
        "step": Ref("step"),
        "step_current_gas": Ref[int]("step.current_gas"),
        "step_inputs": Ref[dict]("step.micro.inputs"),
        "step_should_sync_grad": Ref[bool]("step.should_sync_grad"),
        "step_loss": Ref[torch.Tensor]("step.micro.loss"),
        "step_metrics_history": Ref[dict]("step.metrics_history"),
        "hidden_size": Ref[int](
            "model", key_path=tuple(P.config.text_config.hidden_size)
        ),
        "vocab_size": Ref[int](
            "model", key_path=tuple(P.config.text_config.vocab_size)
        ),
        "dtype": Ref[torch.dtype]("dtype"),
        "empty_steps": Ref[int](
            "args.train.torch_empty_cache_steps", metadata={ARG: Arg(default=1)}
        ),
        "max_grad_norm": Ref[float](
            "args.train.max_grad_norm", metadata={ARG: Arg(default=1.0)}
        ),
        "model_name_or_path": Ref[str](
            "args.model.name_or_path", metadata={ARG: Arg(required=True)}
        ),
        "model_cache_dir": Ref[str](
            "args.model.cache_dir", metadata={ARG: Arg(default=None)}
        ),
        "model_max_length": Ref[int](
            "args.model.max_length", metadata={ARG: Arg(default=16384)}
        ),
        "model_type": Ref[str](
            "args.model.model_type", metadata={ARG: Arg(default="qwen2.5vl")}
        ),
        "data_dataset_use": Ref[str](
            "args.data.dataset_use", metadata={ARG: Arg(required=True)}
        ),
        "data_flatten": Ref[bool](
            "args.data.flatten", metadata={ARG: Arg(default=False)}
        ),
        "data_packing": Ref[bool](
            "args.data.packing", metadata={ARG: Arg(default=False)}
        ),
        "data_base_interval": Ref[int](
            "args.data.base_interval", metadata={ARG: Arg(default=2)}
        ),
        "data_max_pixels": Ref[int](
            "args.data.max_pixels", metadata={ARG: Arg(default=50176)}
        ),
        "data_min_pixels": Ref[int](
            "args.data.min_pixels", metadata={ARG: Arg(default=784)}
        ),
        "data_video_max_frames": Ref[int](
            "args.data.video_max_frames", metadata={ARG: Arg(default=8)}
        ),
        "data_video_min_frames": Ref[int](
            "args.data.video_min_frames", metadata={ARG: Arg(default=4)}
        ),
        "data_video_max_pixels": Ref[int](
            "args.data.video_max_pixels", metadata={ARG: Arg(default=100352)}
        ),
        "data_video_min_pixels": Ref[int](
            "args.data.video_min_pixels", metadata={ARG: Arg(default=50176)}
        ),
        "data_video_fps": Ref[float](
            "args.data.video_fps", metadata={ARG: Arg(default=2)}
        ),
    }


def grpo_scope():
    scope = default_scope()
    new_scope = dict(scope)
    new_scope.update(
        {
            "group_size": Ref[int](
                "args.train.group_size", metadata={ARG: Arg(default=8)}
            ),
            "micro_batch_size": Ref[int](
                "args.train.micro_batch_size", metadata={ARG: Arg(default=4)}
            ),
            "beta": Ref[float]("args.train.beta", metadata={ARG: Arg(default=1e-3)}),
            "rollout_data": Ref[dict]("step.rollout_data"),
            "rewards": Ref[torch.Tensor]("step.rewards"),
            "rewards_dict": Ref("step.rewards_dict"),
            "advantages": Ref[torch.Tensor]("step.advantages"),
            "step_advantages": Ref[torch.Tensor]("step.micro.advantages"),
            "step_diag_accum": Ref[dict]("step.diag_accum"),
            "step_micro_rewards": Ref[torch.Tensor]("step.micro.rewards"),
            "step_micro_rewards_dict": Ref("step.micro.rewards_dict"),
            "step_micro_items": Ref("step.micro.items"),
            "step_micro_inputs": Ref("step.micro.inputs"),
            "step_micro_batches": Ref[list]("step.micro_batches"),
            "reference_model": Ref("reference_model"),
            "inference_engine": Ref("inference_engine"),
            "model_for_generation": Ref("model_for_generation"),
            "rollout_max_new_tokens": Ref[int](
                "args.train.rollout_max_new_tokens", metadata={ARG: Arg(default=30)}
            ),
            "rollout_max_think_tokens": Ref[int](
                "args.train.rollout_max_think_tokens", metadata={ARG: Arg(default=20)}
            ),
            "rollout_temperature": Ref[float](
                "args.train.rollout_temperature", metadata={ARG: Arg(default=1.0)}
            ),
            "rollout_top_k": Ref[int](
                "args.train.rollout_top_k", metadata={ARG: Arg(default=50)}
            ),
            "rollout_top_p": Ref[float](
                "args.train.rollout_top_p", metadata={ARG: Arg(default=0.95)}
            ),
            "rollout_fpc": Ref[float](
                "args.train.rollout_fpc", metadata={ARG: Arg(default=2.0)}
            ),
            "rollout_max_chunks": Ref[int](
                "args.train.rollout_max_chunks",
                metadata={ARG: Arg(default=DEFAULT_MAX_CHUNKS)},
            ),
            "rollout_min_pixels": Ref[int](
                "args.train.rollout_min_pixels",
                metadata={ARG: Arg(default=DEFAULT_INFERENCE_MIN_PIXELS)},
            ),
            "rollout_max_pixels": Ref[int](
                "args.train.rollout_max_pixels",
                metadata={ARG: Arg(default=DEFAULT_INFERENCE_MAX_PIXELS)},
            ),
            "time_reward_window": Ref[int](
                "args.train.time_reward_window", metadata={ARG: Arg(default=5)}
            ),
            "time_reward_slack": Ref[float](
                "args.train.time_reward_slack", metadata={ARG: Arg(default=3.0)}
            ),
            "rollout_last_sync_step": Ref[int]("state.grpo.rollout_last_sync_step"),
        }
    )
    return new_scope
