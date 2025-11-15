import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.shared.array_typing as at


@at.typecheck
class BaseMoEFeedForward(nn.Module):
    expert_dim: int
    hidden_dim: int = 4096
    num_experts: int = 4
    top_k: int = 1

    def setup(self):
        # Router/Gating network
        self.w_gating = self.param(
            "w_gating",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.expert_dim, self.num_experts),
        )

        # Expert hidden and output weights
        self.w_expert_hidden = self.param(
            "w_expert_hidden",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (2, self.num_experts, self.expert_dim, self.hidden_dim),
        )
        self.w_expert_output = self.param(
            "w_expert_output",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.num_experts, self.hidden_dim, self.expert_dim),
        )

    @staticmethod
    def calculate_z_loss(logits, *, lambda_z=1e-4):
        logz = jax.nn.logsumexp(logits, axis=-1)
        return lambda_z * jnp.mean(logz**2)


    def gating_topk(self, x: jnp.ndarray, gate_scale_fn=None):
        B, S, expert_dim = x.shape
        x_flat = x.reshape(-1, expert_dim)
        dtype = x.dtype

        gating_scores = jnp.dot(x_flat, self.w_gating.astype(dtype))
        z_loss = self.calculate_z_loss(gating_scores)
        gating_scores_softmax = nn.softmax(gating_scores, axis=-1)
        mean_gate_score = jnp.mean(gating_scores_softmax, axis=0)

        top_k_values, top_k_indices = jax.lax.top_k(gating_scores_softmax, self.top_k)
        total_tokens = B * S
        one_hot_topk = jax.nn.one_hot(top_k_indices, self.num_experts)
        expert_token_counts = jnp.sum(one_hot_topk, axis=(0, 1))
        activation_ratio = expert_token_counts / total_tokens
        activation_square_sum = jnp.sum(jnp.square(activation_ratio))
        load_balancing_loss = jnp.mean(activation_ratio * mean_gate_score)

        moe_info = {
            "z_loss": z_loss,
            "load_balancing_loss": load_balancing_loss,
            "activation_square_sum": activation_square_sum
        }

        if gate_scale_fn is not None:
            gate_scales = gate_scale_fn(x_flat, gating_scores_softmax, top_k_indices, dtype)
        else:
            gate_scales = top_k_values

        return x_flat, top_k_indices, gate_scales, moe_info


    def forward_expert(self, x_flat, top_k_indices, gate_scales, dtype=jnp.float32):
         # Gate projection
        ff_gate = jnp.einsum("bf, efh -> beh", x_flat, self.w_expert_hidden[0].astype(dtype))
        gate_value = nn.gelu(ff_gate)

        # Up projection
        ff1 = jnp.einsum("bf, efh -> beh", x_flat, self.w_expert_hidden[1].astype(dtype))
        expert_hidden = gate_value * ff1

        # Down projection
        expert_output = jnp.einsum("beh, ehd -> bed", expert_hidden, self.w_expert_output.astype(dtype))

        # Select top-k expert outputs and gate scales
        selected_outputs = jnp.take_along_axis(
            expert_output, top_k_indices[..., None], axis=1
        )  # (B*S, top_k, expert_dim)
        selected_scales = jnp.take_along_axis(
            gate_scales, top_k_indices, axis=1
        )  # (B*S, top_k)

        # Weighted sum
        selected_scales = selected_scales[..., None]  # (B*S, top_k, 1)
        weighted_outputs = jnp.sum(selected_outputs * selected_scales, axis=1)  # (B*S, expert_dim)

        return weighted_outputs



@at.typecheck
class VanillaMoEFeedForward(BaseMoEFeedForward):
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x_flat, top_k_indices, gate_scales, moe_info = self.gating_topk(x)
        weighted_outputs = self.forward_expert(x_flat, top_k_indices, gate_scales)
        output = weighted_outputs.reshape(*x.shape).astype(x.dtype)
        return output, moe_info


@at.typecheck
class AdaMoEFeedForward(BaseMoEFeedForward):
    def setup(self):
        super().setup()
        self.w_gating_scale = self.param(
            "w_gating_scale",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.expert_dim, self.num_experts),
        )

    @nn.compact
    def __call__(self, x: jnp.ndarray):

        def gate_scale_fn(x_flat, gating_scores, top_k_indices, dtype):
            scale = jnp.dot(x_flat, self.w_gating_scale.astype(dtype))
            return gating_scores + scale

        x_flat, top_k_indices, gate_scales, moe_info = self.gating_topk(x, gate_scale_fn=gate_scale_fn)
        weighted_outputs = self.forward_expert(x_flat, top_k_indices, gate_scales)
        output = weighted_outputs.reshape(*x.shape).astype(x.dtype)
        return output, moe_info


@at.typecheck
class CSMoEFeedForward(BaseMoEFeedForward):
    def setup(self):
        super().setup()
        self.w_gating_scale = self.param(
            "w_gating_scale",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.expert_dim + self.num_experts, self.num_experts),
        )

    @nn.compact
    def __call__(self, x: jnp.ndarray):

        def gate_scale_fn(x_flat, gating_scores, top_k_indices, dtype):
            scale_input = jnp.concatenate([x_flat, gating_scores], axis=-1)
            gate_scales = jnp.dot(scale_input, self.w_gating_scale.astype(dtype))
            return nn.softmax(gate_scales, axis=-1)

        x_flat, top_k_indices, gate_scales, moe_info = self.gating_topk(x, gate_scale_fn=gate_scale_fn)
        weighted_outputs = self.forward_expert(x_flat, top_k_indices, gate_scales)
        output = weighted_outputs.reshape(*x.shape).astype(x.dtype)
        return output, moe_info