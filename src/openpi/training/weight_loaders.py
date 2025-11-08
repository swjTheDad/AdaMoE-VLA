import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np
import jax
import jax.numpy as jnp

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(
            download.maybe_download(self.params_path), restore_type=np.ndarray
        )
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz",
            gs={"token": "anon"},
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {
            "PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")[
                "params"
            ]
        }
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(
    loaded_params: at.Params, params: at.Params, *, missing_regex: str
) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype)

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")


@dataclasses.dataclass(frozen=True)
class MoEWeightLoader(WeightLoader):
    params_path: str
    num_experts: int = 4
    noise_std: float = 0.01
    gating_init_std: float = 0.006

    def load(self, params: at.Params) -> at.Params:
        """Load checkpoint and init MoE """
        logger.info(f"[MoEWeightLoader] Start: {self.num_experts} experts")

        raw_params = _model.restore_params(
            download.maybe_download(self.params_path), restore_type=np.ndarray
        )
        flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
        flat_raw = flax.traverse_util.flatten_dict(raw_params, sep="/")

        # --- Merge weights that exist ---
        flat_loaded = {
            k: v.astype(flat_ref[k].dtype)
            for k, v in flat_raw.items()
            if k in flat_ref
        }

        # --- copy action expert weights to new experts ---
        action_expert = self._extract_action_expert(flat_raw)
        if action_expert:
            logger.info(f"Found action expert templates: {list(action_expert.keys())}")
            self._replicate_experts(flat_loaded, flat_ref, action_expert)
        else:
            logger.warning("No action expert weights found for replication")

        # --- init gating network ---
        self._init_gating(flat_loaded, flat_ref)

        missing_count = 0
        for k in flat_ref:
            if k not in flat_loaded:
                flat_loaded[k] = flat_ref[k]
                missing_count += 1
        logger.info(
            f"[MoEWeightLoader] Completed: {missing_count} params filled from reference"
        )        

        logger.info("[MoEWeightLoader] Done.")
        return flax.traverse_util.unflatten_dict(flat_loaded, sep="/")

    def _extract_action_expert(self, flat_raw: dict) -> dict:
        """extract action expert template weights"""
        return {
            k: v
            for k, v in flat_raw.items()
            if any(x in k for x in ["mlp_1/linear", "mlp_1/gating_einsum"])
        }

    def _replicate_experts(self, flat_loaded: dict, flat_ref: dict, action_weights: dict):
        """copy action expert weight to all experts"""
        key = jax.random.PRNGKey(42)

        action_gating = action_weights.get("PaliGemma/llm/layers/mlp_1/gating_einsum", None)
        action_linear = action_weights.get("PaliGemma/llm/layers/mlp_1/linear", None)

        for k, ref_w in flat_ref.items():
            if "moe" not in k or "expert" not in k:
                continue

            # choose template according to weight paths
            if "w_expert_hidden" in k and action_gating is not None:
                template = action_gating
                template = template[:, :, None, :, :]
            elif "w_expert_output" in k and action_linear is not None:
                template = action_linear
                template = template[:, None, :, :]
            else:
                template = None

            # template exist and correct
            if template is not None and template.shape[-2:] == ref_w.shape[-2:]:
                key, subkey = jax.random.split(key)
                noise = jax.random.normal(subkey, ref_w.shape) * self.noise_std
                replicated = jnp.broadcast_to(template, ref_w.shape)
                flat_loaded[k] = (replicated + noise).astype(ref_w.dtype)
            else:
                if template is None:
                    logger.warning(f"No template found for {k}")
                else:
                    logger.warning(
                        f"Shape mismatch for {k}: template{template.shape} vs ref{ref_w.shape}"
                    )

    def _init_gating(self, flat_loaded: dict, flat_ref: dict):
        """initialize gating params"""
        key = jax.random.PRNGKey(0)
        for k, ref_w in flat_ref.items():
            if "w_gating" in k and k not in flat_loaded:
                flat_loaded[k] = (
                    jax.random.normal(key, ref_w.shape) * self.gating_init_std
                ).astype(ref_w.dtype)
