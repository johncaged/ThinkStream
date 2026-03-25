from typing import List

from slyme.utils.registry import Registry
from slyme.context import Ref
from slyme.node import Node
from slyme.builder import builder
from deepslyme.context.metadata import ARG, Arg

# Import deepslyme common nodes
from deepslyme.node.common import (
    set_seed,
    free_memory,
    empty_cache_by_step as empty_cache,
    prepare_distributed_dataloader,
    prepare_inputs,
    clean_step_inputs,
    setup_dtype,
    create_optimizer,
    create_scheduler,
    init_progress,
    update_progress,
    destroy_progress,
    dataloader_loop,
    dataloader_loop_with_micro_steps,
)
from deepslyme.node.integration.deepspeed import (
    deepspeed_init_distributed,
    deepspeed_config_init,
    deepspeed_backward,
    deepspeed_step,
    deepspeed_initialize,
    deepspeed_set_grad_acc_boundary,
    deepspeed_with_grad_acc_boundary,
)
from deepslyme.node.integration.liger_kernel import apply_liger_kernel
from deepslyme.node.sft import (
    init_training_state,
    epoch_loop,
    dataloader_set_epoch,
    compute_loss,
)
from deepslyme.node.metric import collect_metrics, reduce_and_log_metrics
from deepslyme.node.rl.grpo import calc_grpo_advantages

# Import SFT components
from thinkstream.trainer.sft import (
    build_optimizer_kwargs,
    set_gradient_checkpointing,
    align_special_tokens,
    set_model_train,
    model_zero_grad,
    check_should_save,
    update_deepspeed_config_by_hidden_size,
    hf_deepspeed_save_model,
    sft_mini_metrics,
    sft_global_metrics,
    with_hf_deepspeed_context,
    load_model,
    configure_model_gradients,
    init_processor,
    init_dataset,
    train_pipeline,
)
from thinkstream.trainer.scope import default_scope

# Import GRPO components
from thinkstream.trainer.grpo import (
    load_grpo_models,
    unwrap_model_for_generation,
    rollout,
    calc_rewards,
    build_grpo_inputs,
    apply_liger_kernel_for_grpo,
    compute_grpo_loss,
    grpo_micro_metrics,
    grpo_global_metrics,
    prepare_grpo_micro_batches,
    init_grpo_refs,
    init_grpo_dataset,
    timer,
    REWARD_DICT_KEYS,
)
from thinkstream.trainer.scope import grpo_scope


TRAINER_BUILDERS = Registry("TrainerBuilders")


@TRAINER_BUILDERS.register(key="sft")
@builder
def build_hf_deepspeed_train() -> Node:
    scope = default_scope()
    train_data_scope = {
        "dataset": Ref("train_dataset"),
        "train_dataset": Ref("train_dataset"),
        "bsz": Ref(
            "args.train.per_device_train_batch_size", metadata={ARG: Arg(default=1)}
        ),
        "dataloader": Ref("train_dataloader"),
        "train_dataloader": Ref("train_dataloader"),
    }

    train_nodes: list[Node] = [
        deepspeed_init_distributed(scope),
        set_seed(scope),
        deepspeed_config_init(scope, train_data_scope, hidden_size=None),
        setup_dtype(scope),
        build_optimizer_kwargs(scope),
        load_model(scope).add_wrappers(with_hf_deepspeed_context(scope)),
        update_deepspeed_config_by_hidden_size(scope),
        configure_model_gradients(scope),
        init_processor(scope),
        init_dataset(scope, train_data_scope),
        apply_liger_kernel(scope),
        align_special_tokens(scope),
        free_memory(scope),
        prepare_distributed_dataloader(scope, train_data_scope),
        init_training_state(scope, train_data_scope),
        create_optimizer(scope),
        create_scheduler(scope),
        set_gradient_checkpointing(scope),
        set_model_train(scope),
        deepspeed_initialize(scope),
        model_zero_grad(scope),
        init_progress(scope),
        epoch_loop(
            scope,
            nodes=[
                dataloader_set_epoch(scope, train_data_scope),
                dataloader_loop(
                    scope,
                    train_data_scope,
                    mini_step_nodes=[
                        deepspeed_set_grad_acc_boundary(scope),
                        prepare_inputs(scope),
                        compute_loss(scope),
                        collect_metrics(scope, current_metrics=sft_mini_metrics(scope)),
                        clean_step_inputs(scope),
                        empty_cache(scope),
                        deepspeed_backward(scope),
                    ],
                    global_step_nodes=[
                        deepspeed_step(scope).add_wrappers(
                            deepspeed_with_grad_acc_boundary(
                                scope,
                                step_should_sync_grad=True,
                                reset_boundary_to=False,
                            )
                        ),
                        collect_metrics(
                            scope, current_metrics=sft_global_metrics(scope)
                        ),
                        reduce_and_log_metrics(
                            scope,
                            metric_defs={
                                "loss": "mean",
                                "grad_norm": "latest",
                                "learning_rate": "latest",
                            },
                        ),
                        update_progress(scope),
                        hf_deepspeed_save_model(scope).add_wrappers(
                            check_should_save(scope)
                        ),
                    ],
                ),
            ],
        ),
        hf_deepspeed_save_model(scope),  # Always save
        destroy_progress(scope),
    ]
    return train_pipeline(nodes=train_nodes)


@TRAINER_BUILDERS.register(key="grpo")
@builder
def build_grpo_train() -> Node:
    scope = grpo_scope()
    train_data_scope = {
        "dataset": Ref("train_dataset"),
        "train_dataset": Ref("train_dataset"),
        "bsz": Ref(
            "args.train.per_device_train_batch_size", metadata={ARG: Arg(default=1)}
        ),
        "dataloader": Ref("train_dataloader"),
        "train_dataloader": Ref("train_dataloader"),
    }

    grpo_nodes: List[Node] = [
        deepspeed_init_distributed(scope),
        set_seed(scope),
        init_grpo_refs(scope),
        deepspeed_config_init(scope, train_data_scope, hidden_size=None),
        setup_dtype(scope),
        build_optimizer_kwargs(scope),
        load_grpo_models(scope),
        update_deepspeed_config_by_hidden_size(scope),
        init_processor(scope),
        init_grpo_dataset(scope, train_data_scope),
        apply_liger_kernel_for_grpo(scope),
        align_special_tokens(scope),
        free_memory(scope),
        prepare_distributed_dataloader(scope, train_data_scope),
        init_training_state(scope, train_data_scope),
        create_optimizer(scope),
        create_scheduler(scope),
        set_gradient_checkpointing(scope),
        set_model_train(scope),
        deepspeed_initialize(scope),
        model_zero_grad(scope),
        init_progress(scope),
        epoch_loop(
            scope,
            nodes=[
                dataloader_set_epoch(scope, train_data_scope),
                dataloader_loop_with_micro_steps(
                    scope,
                    train_data_scope,
                    mini_step_nodes=[
                        rollout(scope).add_wrappers(
                            timer(name="Sync + Rollout"),
                            unwrap_model_for_generation(scope),
                        ),
                        calc_rewards(scope),
                        calc_grpo_advantages(scope),
                        prepare_grpo_micro_batches(scope),
                    ],
                    micro_step_nodes=[
                        deepspeed_set_grad_acc_boundary(scope),
                        build_grpo_inputs(scope),
                        prepare_inputs(scope, step_inputs=scope["step_micro_inputs"]),
                        compute_grpo_loss(scope).add_wrappers(
                            timer(name="Compute GRPO Loss")
                        ),
                        clean_step_inputs(
                            scope, step_inputs=scope["step_micro_inputs"]
                        ),
                        empty_cache(scope),
                        deepspeed_backward(scope).add_wrappers(timer(name="Backward")),
                        collect_metrics(
                            scope, current_metrics=grpo_micro_metrics(scope)
                        ),
                    ],
                    global_step_nodes=[
                        deepspeed_step(scope).add_wrappers(
                            deepspeed_with_grad_acc_boundary(
                                scope,
                                step_should_sync_grad=True,
                                reset_boundary_to=False,
                            )
                        ),
                        collect_metrics(
                            scope, current_metrics=grpo_global_metrics(scope)
                        ),
                        reduce_and_log_metrics(
                            scope,
                            metric_defs={
                                "loss": "mean",
                                "reward_mean": "mean",
                                "reward_var": "mean",
                                "avg_think_len": "mean",
                                "grad_norm": "latest",
                                "learning_rate": "latest",
                                **{
                                    f"reward_{k}_mean": "mean" for k in REWARD_DICT_KEYS
                                },
                            },
                        ),
                        update_progress(scope),
                        hf_deepspeed_save_model(scope).add_wrappers(
                            check_should_save(scope)
                        ),
                    ],
                ),
            ],
        ),
        hf_deepspeed_save_model(scope),
        destroy_progress(scope),
    ]

    return train_pipeline(nodes=grpo_nodes)
