from copy import deepcopy

import jax
import jax.numpy as jnp

from flax import linen as nn

from transformers.models.roberta.modeling_flax_roberta import (
    FlaxRobertaModule, RobertaConfig, FlaxRobertaPreTrainedModel, FlaxRobertaClassificationHead
)


class CustomFlaxRobertaClassificationHead(nn.Module):
    config: RobertaConfig
    dtype: jnp.dtype = jnp.float32
    use_bias: bool = True

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )
        classifier_dropout = (
            self.config.classifier_dropout
            if self.config.classifier_dropout is not None
            else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(rate=classifier_dropout)
        self.out_proj = nn.Dense(
            self.config.num_labels,
            use_bias=self.use_bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

    def __call__(self, hidden_states, deterministic=True):
        # commented out to have a classification head comparable to the Wrench benchmark

        hidden_states = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS])
        # hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        # hidden_states = self.dense(hidden_states)
        # hidden_states = nn.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class SepLLModule(nn.Module):
    config: RobertaConfig
    T: jnp.array
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # std roberta
        self.roberta = FlaxRobertaModule(config=self.config, dtype=self.dtype, add_pooling_layer=False)
        self.config.num_labels = self.T.shape[1]
        self.classifier = FlaxRobertaClassificationHead(config=self.config, dtype=self.dtype)

        # lf path
        self.lf_config = deepcopy(self.config)
        self.lf_config.num_labels = self.T.shape[0]
        self.lf_classifier = CustomFlaxRobertaClassificationHead(config=self.lf_config, use_bias=True, dtype=self.dtype)

    def __call__(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            deterministic: bool = True,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        # Model
        outputs = self.roberta(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        logits = self.classifier(sequence_output, deterministic=deterministic)
        lf_logits = self.lf_classifier(sequence_output, deterministic=deterministic)

        w = jnp.dot(logits, jnp.transpose(self.T)) + lf_logits

        return nn.softmax(logits, axis=-1), lf_logits, w


class SepLLModel(FlaxRobertaPreTrainedModel):
    module_class = SepLLModule
