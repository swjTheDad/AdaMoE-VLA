import dataclasses
from typing import Protocol, runtime_checkable

import jax.numpy as jnp
import optax

import openpi.shared.array_typing as at


@runtime_checkable
class LRScheduleConfig(Protocol):

    def create(self) -> optax.Schedule: ...


@dataclasses.dataclass(frozen=True)
class CosineDecaySchedule(LRScheduleConfig):
    """Cosine decay schedule with warmup."""

    warmup_steps: int = 1000
    peak_lr: float = 2.5e-5
    decay_steps: int = 30_000
    decay_lr: float = 2.5e-6

    def create(self) -> optax.Schedule:
        return optax.warmup_cosine_decay_schedule(
            init_value=self.peak_lr / (self.warmup_steps + 1),
            peak_value=self.peak_lr,
            warmup_steps=self.warmup_steps,
            decay_steps=self.decay_steps,
            end_value=self.decay_lr,
        )


@dataclasses.dataclass(frozen=True)
class RsqrtDecaySchedule(LRScheduleConfig):
    """Inverse square root decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 5e-5
    timescale: float = 10_000

    def create(self) -> optax.Schedule:
        return optax.join_schedules(
            [
                optax.linear_schedule(
                    init_value=self.peak_lr / (self.warmup_steps + 1),
                    end_value=self.peak_lr,
                    transition_steps=self.warmup_steps,
                ),
                lambda step: self.peak_lr
                / jnp.sqrt((self.timescale + step) / self.timescale),
            ],
            [self.warmup_steps],
        )


@runtime_checkable
class OptimizerConfig(Protocol):

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation: ...


@dataclasses.dataclass(frozen=True)
class AdamW(OptimizerConfig):
    """AdamW optimizer."""

    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 1e-10
    clip_gradient_norm: float = 1.0

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        tx = optax.adamw(
            lr,
            b1=self.b1,
            b2=self.b2,
            eps=self.eps,
            weight_decay=self.weight_decay,
            mask=weight_decay_mask,
        )

        return optax.chain(optax.clip_by_global_norm(self.clip_gradient_norm), tx)


@dataclasses.dataclass(frozen=True)
class SGD(OptimizerConfig):
    """SGD optimizer."""

    lr: float = 5e-5
    momentum: float = 0.9
    nesterov: bool = False

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        assert weight_decay_mask is None, "Weight decay is not supported for SGD"
        return optax.sgd(lr, momentum=self.momentum, nesterov=self.nesterov)


@dataclasses.dataclass(frozen=True)
class MultiGroupAdamW(OptimizerConfig):
    """Multi-group AdamW optimizer with different learning rates and schedules for different parameter groups."""

    # Base optimizer settings (for "base" group)
    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    clip_gradient_norm: float = 1.0

    # Learning rates for different groups (peak values for cosine decay)
    lr_base: float = 2.5e-4
    lr_moe: float = 2.5e-4
    lr_router: float = 2.5e-3

    # Weight decay for different groups
    wd_base: float = 1e-10
    wd_moe: float = 1e-10
    wd_router: float = 1e-4

    # Cosine decay schedule parameters for each group
    # Warmup steps
    warmup_steps_base: int = 1000
    warmup_steps_moe: int = 1000
    warmup_steps_router: int = 1000
    # Decay steps
    decay_steps_base: int = 30_000
    decay_steps_moe: int = 30_000
    decay_steps_router: int = 30_000
    # Final learning rates
    decay_lr_base: float = 5e-6
    decay_lr_moe: float = 5e-6
    decay_lr_router: float = 2.5e-4

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        """Create multi-group optimizer using official optax.partition.

        Note: The lr parameter is ignored since we use group-specific learning rate schedules.
        """

        # Create individual cosine decay schedules for each group
        def create_cosine_schedule(warmup_steps, peak_lr, decay_steps, decay_lr):
            return optax.warmup_cosine_decay_schedule(
                init_value=peak_lr / (warmup_steps + 1),
                peak_value=peak_lr,
                warmup_steps=warmup_steps,
                decay_steps=decay_steps,
                end_value=decay_lr,
            )

        schedules = {
            "base": create_cosine_schedule(
                self.warmup_steps_base,
                self.lr_base,
                self.decay_steps_base,
                self.decay_lr_base,
            ),
            "moe": create_cosine_schedule(
                self.warmup_steps_moe,
                self.lr_moe,
                self.decay_steps_moe,
                self.decay_lr_moe,
            ),
            "router": create_cosine_schedule(
                self.warmup_steps_router,
                self.lr_router,
                self.decay_steps_router,
                self.decay_lr_router,
            ),
        }

        # Print optimizer configuration for each group
        print("=" * 80)
        print("🔧 MULTI-GROUP ADAMW OPTIMIZER WITH COSINE DECAY SCHEDULES")
        print("=" * 80)

        # Create transforms for each group with individual schedules
        transforms = {
            "base": optax.adamw(
                learning_rate=schedules["base"],
                b1=self.b1,
                b2=self.b2,
                eps=self.eps,
                weight_decay=self.wd_base,
            ),
            "moe": optax.adamw(
                learning_rate=schedules["moe"],
                b1=self.b1,
                b2=self.b2,
                eps=self.eps,
                weight_decay=self.wd_moe,
            ),
            "router": optax.adamw(
                learning_rate=schedules["router"],
                b1=self.b1,
                b2=self.b2,
                eps=self.eps,
                weight_decay=self.wd_router,
            ),
        }

        # Print configuration for each group with schedule details
        group_configs = [
            (
                "base",
                "Base model components",
                self.lr_base,
                self.wd_base,
                self.warmup_steps_base,
                self.decay_steps_base,
                self.decay_lr_base,
            ),
            (
                "moe",
                "Mixture of Experts parameters",
                self.lr_moe,
                self.wd_moe,
                self.warmup_steps_moe,
                self.decay_steps_moe,
                self.decay_lr_moe,
            ),
            (
                "router",
                "Router expert layers",
                self.lr_router,
                self.wd_router,
                self.warmup_steps_router,
                self.decay_steps_router,
                self.decay_lr_router,
            ),
        ]

        for (
            group_name,
            description,
            peak_lr,
            wd,
            warmup,
            decay_steps,
            final_lr,
        ) in group_configs:
            print(f"📋 {group_name:>8} group: {description}")
            print(f"   └─ Peak LR:        {peak_lr:.2e}")
            print(f"   └─ Final LR:       {final_lr:.2e}")
            print(f"   └─ Warmup steps:   {warmup:,}")
            print(f"   └─ Decay steps:    {decay_steps:,}")
            print(f"   └─ Weight Decay:   {wd:.2e}")
            print(f"   └─ b1={self.b1}, b2={self.b2}, eps={self.eps}")
            print()

        print(f"🔗 Gradient Clipping: {self.clip_gradient_norm}")
        print("=" * 80)

        def classify_param_by_path(path):
            """Classify parameter into groups based on full parameter path."""
            # Apply classification logic to the full path
            if "moe" in path and "gating" in path:
                # When both 'moe' and 'gating' are in path, use router group
                return "router"
            elif "moe" in path:
                return "moe"
            else:
                return "base"

        def create_path_aware_label_fn():
            """Create a label function that considers full parameter paths."""

            def label_fn(params_dict):
                """Generate labels based on full parameter paths."""

                def create_labels_tree(params_tree, path=""):
                    """Recursively create labels tree matching params structure."""
                    if isinstance(params_tree, dict):
                        labels = {}
                        for key, value in params_tree.items():
                            current_path = f"{path}/{key}" if path else key
                            labels[key] = create_labels_tree(value, current_path)
                        return labels
                    else:
                        # This is a leaf (actual parameter) - classify based on full path
                        return classify_param_by_path(path)

                return create_labels_tree(params_dict)

            return label_fn

        # Create the label function that considers full parameter paths
        label_fn = create_path_aware_label_fn()

        # Create a wrapper that handles nnx.State conversion
        def state_aware_partition(transforms, label_fn):
            """Wrapper for optax.partition that handles nnx.State objects."""
            base_partition = optax.partition(transforms, label_fn)

            def init_fn(params):
                # Convert State to pure dict for optax.partition
                if hasattr(params, "to_pure_dict"):
                    params_dict = params.to_pure_dict()
                else:
                    params_dict = params

                # Print parameter grouping information
                print("\n🏷️ PARAMETER GROUP ASSIGNMENTS:")
                print("=" * 60)

                # Get labels for all parameters to show grouping
                param_labels = label_fn(params_dict)
                group_counts = {}
                group_params = {}

                def collect_group_info(labels_tree, params_tree, path=""):
                    """Recursively collect parameter group information."""
                    if isinstance(labels_tree, dict) and isinstance(params_tree, dict):
                        for key in labels_tree:
                            if key in params_tree:
                                current_path = f"{path}/{key}" if path else key
                                collect_group_info(
                                    labels_tree[key], params_tree[key], current_path
                                )
                    else:
                        # This is a leaf - labels_tree is the group name
                        group = labels_tree
                        if group not in group_counts:
                            group_counts[group] = 0
                            group_params[group] = []
                        group_counts[group] += 1
                        group_params[group].append(path)

                collect_group_info(param_labels, params_dict)

                # Print summary
                total_params = sum(group_counts.values())
                for group in ["base", "moe", "router"]:
                    if group in group_counts:
                        count = group_counts[group]
                        percentage = (count / total_params) * 100
                        print(
                            f"📦 {group:>8} group: {count:3d} parameters ({percentage:5.1f}%)"
                        )

                        # Show first few parameter paths as examples
                        example_paths = group_params[group][:3]
                        for path in example_paths:
                            print(f"   └─ {path}")
                        if len(group_params[group]) > 3:
                            print(f"   └─ ... and {len(group_params[group]) - 3} more")
                        print()

                print(f"📊 Total parameters: {total_params}")
                print("=" * 60)

                return base_partition.init(params_dict)

            def update_fn(updates, state, params=None):
                # Convert inputs to pure dicts
                if hasattr(updates, "to_pure_dict"):
                    updates_dict = updates.to_pure_dict()
                    is_state_updates = True
                else:
                    updates_dict = updates
                    is_state_updates = False

                if params is not None and hasattr(params, "to_pure_dict"):
                    params_dict = params.to_pure_dict()
                else:
                    params_dict = params

                # Apply the base partition update
                result_updates, new_state = base_partition.update(
                    updates_dict, state, params_dict
                )

                # Convert result back to State if input was State
                if is_state_updates:
                    import flax.nnx as nnx

                    # Create a new State with the same structure as original
                    import copy

                    result_state = copy.deepcopy(updates)
                    result_state.replace_by_pure_dict(result_updates)
                    result_updates = result_state

                return result_updates, new_state

            return optax.GradientTransformation(init_fn, update_fn)

        # Create the partitioned optimizer with State compatibility
        multi_tx = state_aware_partition(transforms, label_fn)

        # Add gradient clipping
        return optax.chain(optax.clip_by_global_norm(self.clip_gradient_norm), multi_tx)


def create_optimizer(
    optimizer: OptimizerConfig,
    lr_schedule: LRScheduleConfig,
    weight_decay_mask: at.PyTree | None = None,
) -> optax.GradientTransformation:
    lr = lr_schedule.create()
    return optimizer.create(lr, weight_decay_mask=weight_decay_mask)
