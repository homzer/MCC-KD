import collections
import copy
import math

import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    RowParallelLinear,
    ColumnParallelLinear,
    ParallelEmbedding
)
import torch
import torch.nn as nn
import torch.nn.init as init
from transformers.activations import ACT2FN
from transformers.models.t5.modeling_t5 import T5LayerNorm

from src.models.modeling import ParallelModelForSeq2SeqLM, Seq2SeqLMOutputs
from src.models.modeling_args import T5Config, LoraT5Config
from src.utils import set_barrier, apply_lora


def _relative_position_bucket(_relative_position, _bidirectional=True, _num_buckets=32, _max_distance=128):
    relative_buckets = 0
    if _bidirectional:
        _num_buckets //= 2
        relative_buckets += (_relative_position > 0).to(torch.long) * _num_buckets
        _relative_position = torch.abs(_relative_position)
    else:
        _relative_position = -torch.min(_relative_position, torch.zeros_like(_relative_position))
    # now relative_position is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = _num_buckets // 2
    is_small = _relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    relative_position_if_large = max_exact + (
            torch.log(_relative_position.float() / max_exact)
            / math.log(_max_distance / max_exact)
            * (_num_buckets - max_exact)
    ).to(torch.long)
    relative_position_if_large = torch.min(
        relative_position_if_large, torch.full_like(relative_position_if_large, _num_buckets - 1)
    )

    relative_buckets += torch.where(is_small, _relative_position, relative_position_if_large)
    return relative_buckets


def compute_relative_position_bias(
        query_length: int,
        key_length: int,
        bidirectional: bool,
        relative_attention_bias: torch.nn.Module,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128
):
    """
    Compute binned relative position bias
    :param relative_attention_num_buckets:
    :param relative_attention_max_distance:
    :param query_length:
    :param key_length:
    :param bidirectional: True if is encoder, False if is decoder
    :param relative_attention_bias: torch.nn.Module get the parameters of position embeddings.
    :return:
    """
    context_position = torch.arange(query_length, dtype=torch.long)[:, None]
    memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
    relative_position = memory_position - context_position  # shape (query_length, key_length)
    relative_position_bucket = _relative_position_bucket(  # type: torch.Tensor
        relative_position,  # shape (query_length, key_length)
        _bidirectional=bidirectional,
        _num_buckets=relative_attention_num_buckets,
        _max_distance=relative_attention_max_distance,
    )
    values = relative_attention_bias(
        relative_position_bucket.to(
            relative_attention_bias.weight.device
        ))  # shape (query_length, key_length, num_heads)
    values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
    return values


def get_decoder_attention_mask(query_length, key_length, start_pos=None):
    """ Causal Attention Mask. """
    start_pos = 0 if start_pos is None else start_pos
    attention_mask = None
    if query_length > 1:
        attention_mask = torch.full(
            (1, 1, query_length, key_length), float("-inf"))
        attention_mask = torch.triu(attention_mask, diagonal=start_pos + 1)
    return attention_mask


def get_encoder_attention_mask(input_ids):
    input_mask = (input_ids == 0).float()  # [b, s]
    attention_mask = torch.ones_like(input_mask)[:, None, :, None] @ input_mask[:, None, None, :]
    return torch.masked_fill(attention_mask, attention_mask.bool(), float("-inf"))


def get_cross_attention_mask(encoder_attention_mask, query_length):
    attention_mask = encoder_attention_mask[:, :, 0, :].unsqueeze(dim=2)
    return torch.ones(1, 1, query_length, 1).to(attention_mask.device) * attention_mask


class ParallelT5Attention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        self.config = config
        self.is_decoder = config.is_decoder
        self.max_seq_len = config.max_output_len if self.is_decoder else config.max_input_len
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        assert config.num_heads % fs_init.get_model_parallel_world_size() == 0
        self.n_heads = config.num_heads // fs_init.get_model_parallel_world_size()
        self.inner_dim = self.n_heads * config.d_kv

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads)

        self.q = None
        self.k = None
        self.v = None
        self.o = None

        self.dropout = nn.Dropout(config.dropout_rate)

        self.cache_k = None
        self.cache_v = None

    def init_weights(self):
        self.q = ColumnParallelLinear(
            self.d_model,
            self.config.num_heads * self.config.d_kv,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.k = ColumnParallelLinear(
            self.d_model,
            self.config.num_heads * self.config.d_kv,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.v = ColumnParallelLinear(
            self.d_model,
            self.config.num_heads * self.config.d_kv,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.o = RowParallelLinear(
            self.config.num_heads * self.config.d_kv,
            self.d_model,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

    def reshape(self, hidden_states):
        batch_size = hidden_states.shape[0]
        return hidden_states.view(
            batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(2, 1)

    def get_key_value_states(self, hidden_states, encoder_hidden_states, use_cache, start):
        if not use_cache:
            if encoder_hidden_states is None:  # Self Attn
                key_states = self.reshape(self.k(hidden_states))
                value_states = self.reshape(self.v(hidden_states))
            else:  # Cross Attn
                key_states = self.reshape(self.k(encoder_hidden_states))
                value_states = self.reshape(self.v(encoder_hidden_states))
        else:
            assert self.is_decoder, "Only decoder could use cache."
            batch_size, seq_len = hidden_states.shape[:2]

            if encoder_hidden_states is None:  # Cached Self Attn
                if self.cache_k is None:
                    self.cache_k = torch.zeros(
                        (batch_size, self.n_heads, self.max_seq_len, self.key_value_proj_dim)
                    ).to(hidden_states)
                if self.cache_v is None:
                    self.cache_v = torch.zeros(
                        (batch_size, self.n_heads, self.max_seq_len, self.key_value_proj_dim)
                    ).to(hidden_states)
                self.cache_k[:batch_size, :, start: start + seq_len, :] = self.reshape(self.k(hidden_states))
                self.cache_v[:batch_size, :, start: start + seq_len, :] = self.reshape(self.v(hidden_states))
                key_states = self.cache_k[:batch_size, :, : start + seq_len, :]
                value_states = self.cache_v[:batch_size, :, : start + seq_len, :]
            else:  # Cached Cross Attn
                if self.cache_k is None:
                    self.cache_k = self.reshape(self.k(encoder_hidden_states))
                if self.cache_v is None:
                    self.cache_v = self.reshape(self.v(encoder_hidden_states))
                key_states = self.cache_k
                value_states = self.cache_v
        return key_states, value_states

    def get_position_bias(self, seq_length, use_cache, start_pos, encoder_seq_length=None):
        query_length = start_pos + seq_length if use_cache else seq_length
        key_length = query_length if encoder_seq_length is None else encoder_seq_length

        if self.has_relative_attention_bias:
            position_bias = compute_relative_position_bias(
                query_length=query_length,
                key_length=key_length,
                bidirectional=(not self.is_decoder),
                relative_attention_bias=self.relative_attention_bias,
                relative_attention_num_buckets=self.relative_attention_num_buckets,
                relative_attention_max_distance=self.relative_attention_max_distance
            )
        else:
            position_bias = torch.zeros((1, self.n_heads, query_length, key_length))

        if use_cache:
            position_bias = position_bias[:, :, -seq_length:, :]
        return position_bias

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            position_bias: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            use_cache: bool = False,
            start_pos: int = None
    ):

        query_states = self.reshape(self.q(hidden_states))
        key_states, value_states = self.get_key_value_states(
            hidden_states, encoder_hidden_states, use_cache, start_pos)
        scores = torch.matmul(query_states, key_states.transpose(-1, -2))

        if position_bias is None:
            position_bias = self.get_position_bias(
                hidden_states.shape[1], use_cache, start_pos,
                encoder_seq_length=None if (
                        encoder_hidden_states is None
                ) else encoder_hidden_states.shape[1]
            )
        if attention_mask is not None:
            position_bias = position_bias + attention_mask.to(position_bias.device)

        scores = scores + position_bias.to(scores)
        attn_weights = torch.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = self.dropout.forward(attn_weights)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = torch.reshape(
            torch.transpose(
                attn_output, 1, 2
            ), shape=[hidden_states.shape[0], -1, self.inner_dim]
        )
        attn_output = self.o(attn_output)
        Outputs = collections.namedtuple('Outputs', ['hidden_states', 'position_bias'])
        return Outputs(hidden_states=attn_output, position_bias=position_bias)

    def flush(self):
        """ Clean self.cache for next inference. """
        self.cache_v = None
        self.cache_k = None


class ParallelT5Feedforward(nn.Module):
    def __init__(self, config: T5Config):
        super().__init__()
        self.config = config
        self.wi_0 = None
        self.wi_1 = None
        self.wo = None
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = None

    def init_weights(self):
        self.wi_0 = ColumnParallelLinear(
            self.config.d_model, self.config.d_ff, bias=False, gather_output=False,
        )
        self.wi_1 = ColumnParallelLinear(
            self.config.d_model, self.config.d_ff, bias=False, gather_output=False,
        )
        self.wo = RowParallelLinear(
            self.config.d_ff, self.config.d_model, bias=False, input_is_parallel=True,
        )
        self.act = ACT2FN[self.config.dense_act_fn]

    def forward(self, hidden_states):
        hidden_gelu = self.act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class ParallelT5Block(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()
        self.config = config
        self.is_decoder = config.is_decoder
        # self attention
        self.SelfAttention = ParallelT5Attention(config, has_relative_attention_bias)
        self.self_attn_layer_norm = None
        self.self_attn_dropout = nn.Dropout(config.dropout_rate)

        # cross attention
        if self.is_decoder:
            self.CrossAttention = ParallelT5Attention(config)
            self.cross_attn_layer_norm = None
            self.cross_attn_dropout = nn.Dropout(config.dropout_rate)

        # feed forward
        self.Feedforward = ParallelT5Feedforward(config)
        self.ffn_layer_norm = None
        self.ffn_dropout = nn.Dropout(config.dropout_rate)

    def init_weights(self):
        self.SelfAttention.init_weights()
        self.self_attn_layer_norm = T5LayerNorm(self.config.d_model, eps=self.config.layer_norm_epsilon)

        if self.is_decoder:
            self.CrossAttention.init_weights()
            self.cross_attn_layer_norm = T5LayerNorm(self.config.d_model, eps=self.config.layer_norm_epsilon)

        self.Feedforward.init_weigths()
        self.ffn_layer_norm = T5LayerNorm(self.config.d_model, eps=self.config.layer_norm_epsilon)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_bias=None,
            encoder_hidden_states=None,
            cross_attention_mask=None,
            encoder_decoder_position_bias=None,
            use_cache=False,
            start_pos=None
    ):
        normed_hidden_states = self.self_attn_layer_norm.forward(hidden_states)
        self_attn_outputs = self.SelfAttention.forward(
            normed_hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            use_cache=use_cache,
            start_pos=start_pos
        )
        hidden_states = hidden_states + self.self_attn_dropout.forward(
            self_attn_outputs.hidden_states
        )

        if self.is_decoder:
            normed_hidden_states = self.cross_attn_layer_norm.forward(hidden_states)
            cross_attn_outputs = self.CrossAttention.forward(
                normed_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=cross_attention_mask,
                position_bias=encoder_decoder_position_bias,
                use_cache=use_cache,
                start_pos=start_pos
            )
            encoder_decoder_position_bias = cross_attn_outputs.position_bias
            hidden_states = hidden_states + self.cross_attn_dropout.forward(
                cross_attn_outputs.hidden_states
            )

        forwarded_states = self.ffn_layer_norm.forward(hidden_states)
        forwarded_states = self.Feedforward.forward(forwarded_states)
        hidden_states = hidden_states + self.ffn_dropout.forward(forwarded_states)

        Outputs = collections.namedtuple('Outputs', [
            'hidden_states', 'position_bias', "encoder_decoder_position_bias"])
        return Outputs(
            hidden_states=hidden_states,
            position_bias=self_attn_outputs.position_bias,
            encoder_decoder_position_bias=encoder_decoder_position_bias
        )


class ParallelT5Stack(nn.Module):
    def __init__(self, config: T5Config, embed_tokens=None):
        super().__init__()
        self.config = config
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        self.num_layers = config.num_layers
        self.block = nn.ModuleList(
            [ParallelT5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = None
        self.dropout = nn.Dropout(config.dropout_rate)

    def init_weights(self):
        for block in self.block:
            block.init_weights()
        self.final_layer_norm = T5LayerNorm(self.config.d_model, eps=self.config.layer_norm_epsilon)

    def forward(
            self,
            input_ids,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            use_cache=False,
            start_pos=None
    ):
        inputs_embeds = self.embed_tokens.forward(input_ids.to(next(self.parameters()).device))
        hidden_states = self.dropout.forward(inputs_embeds)
        if self.is_decoder:
            seq_length = input_ids.shape[1]
            attention_mask = get_decoder_attention_mask(
                query_length=seq_length,
                key_length=seq_length,
                start_pos=start_pos
            )
            assert encoder_attention_mask is not None
            cross_attention_mask = get_cross_attention_mask(
                encoder_attention_mask=encoder_attention_mask,
                query_length=seq_length
            )
        else:
            attention_mask = get_encoder_attention_mask(input_ids)
            cross_attention_mask = None
        position_bias = None
        encoder_decoder_position_bias = None
        for i in range(self.num_layers):
            layer_outputs = self.block[i].forward(
                hidden_states,
                attention_mask=attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_mask=cross_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                use_cache=use_cache,
                start_pos=start_pos
            )
            hidden_states = layer_outputs.hidden_states
            position_bias = layer_outputs.position_bias
            encoder_decoder_position_bias = layer_outputs.encoder_decoder_position_bias

        hidden_states = self.final_layer_norm.forward(hidden_states)
        hidden_states = self.dropout.forward(hidden_states)
        Outputs = collections.namedtuple('Outputs', ['last_hidden_states'])
        return Outputs(last_hidden_states=hidden_states)


class ParallelT5ForConditionalGeneration(ParallelModelForSeq2SeqLM):
    def __init__(self, config: T5Config, local_rank, world_size):
        super().__init__(local_rank, world_size)
        self.config = config
        self.model_dim = config.d_model
        self.shared = None
        self.encoder = None
        self.decoder = None
        self.lm_head = None

    def init_weights(self):
        self.shared = ParallelEmbedding(self.config.vocab_size, self.config.d_model)

        encoder_config = copy.deepcopy(self.config)
        encoder_config.is_decoder = False
        self.encoder = ParallelT5Stack(encoder_config, self.shared)
        self.encoder.init_weights()

        decoder_config = copy.deepcopy(self.config)
        decoder_config.is_decoder = True
        decoder_config.num_layers = self.config.num_decoder_layers
        self.decoder = ParallelT5Stack(decoder_config, self.shared)
        self.decoder.init_weights()

        self.lm_head = ColumnParallelLinear(
            self.config.d_model, self.config.vocab_size, bias=False, gather_output=True,
        )

    def forward(
            self,
            input_ids,
            decoder_input_ids,
            encoder_hidden_states=None,
            use_cache=False,
            start_pos=None
    ):
        if encoder_hidden_states is None:
            encoder_hidden_states = self.encoder.forward(
                input_ids=input_ids
            ).last_hidden_states

        decoder_outputs = self.decoder.forward(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=get_encoder_attention_mask(input_ids),
            use_cache=use_cache,
            start_pos=start_pos
        )
        sequence_output = decoder_outputs.last_hidden_states

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head.forward(sequence_output)
        return Seq2SeqLMOutputs(logits=lm_logits, encoder_hidden_states=encoder_hidden_states)

    def flush(self):
        """ Cleaning cache in `Attention` module """
        for i in range(self.decoder.num_layers):
            self.decoder.block[i].SelfAttention.flush()
            self.decoder.block[i].CrossAttention.flush()
        set_barrier()


class LoraParallelT5Attention(ParallelT5Attention):
    def __init__(self, config: LoraT5Config, has_relative_attention_bias=False):
        super().__init__(config, has_relative_attention_bias)
        self.config = config
        self.lora_a_q = None
        self.lora_b_q = None
        self.lora_a_k = None
        self.lora_b_k = None
        self.lora_a_v = None
        self.lora_b_v = None
        self.lora_a_o = None
        self.lora_b_o = None

    def init_weights(self):
        super().init_weights()
        self.lora_a_q = nn.Linear(
            self.config.d_model, self.config.r, bias=False
        ).type(self.config.lora_dtype)
        self.lora_b_q = ColumnParallelLinear(
            self.config.r,
            self.config.num_heads * self.config.d_kv,
            bias=False,
            gather_output=False,
            init_method=init.zeros_
        ).type(self.config.lora_dtype)

        self.lora_a_k = nn.Linear(
            self.config.d_model, self.config.r, bias=False
        ).type(self.config.lora_dtype)
        self.lora_b_k = ColumnParallelLinear(
            self.config.r,
            self.config.num_heads * self.config.d_kv,
            bias=False,
            gather_output=False,
            init_method=init.zeros_
        ).type(self.config.lora_dtype)

        self.lora_a_v = nn.Linear(
            self.config.d_model, self.config.r, bias=False
        ).type(self.config.lora_dtype)
        self.lora_b_v = ColumnParallelLinear(
            self.config.r,
            self.config.num_heads * self.config.d_kv,
            bias=False,
            gather_output=False,
            init_method=init.zeros_
        ).type(self.config.lora_dtype)

        self.lora_a_o = RowParallelLinear(
            self.config.num_heads * self.config.d_kv,
            self.config.r,
            bias=False,
            input_is_parallel=True,
            init_method=init.xavier_normal_
        ).type(self.config.lora_dtype)
        self.lora_b_o = nn.Linear(
            self.config.r, self.config.d_model, bias=False
        ).type(self.config.lora_dtype)
        init.zeros_(self.lora_b_o.weight)

    def get_key_value_states(self, hidden_states, encoder_hidden_states, use_cache, start):
        if not use_cache:
            if encoder_hidden_states is None:  # Self Attn
                key_states = self.reshape(
                    self.k(hidden_states) + apply_lora(hidden_states, self.lora_a_k, self.lora_b_k)
                )
                value_states = self.reshape(
                    self.v(hidden_states) + apply_lora(hidden_states, self.lora_a_v, self.lora_b_v)
                )
            else:  # Cross Attn
                key_states = self.reshape(
                    self.k(encoder_hidden_states) + apply_lora(encoder_hidden_states, self.lora_a_k, self.lora_b_k)
                )
                value_states = self.reshape(
                    self.v(encoder_hidden_states) + apply_lora(encoder_hidden_states, self.lora_a_v, self.lora_b_v)
                )
        else:
            assert self.is_decoder, "Only decoder could use cache."
            batch_size, seq_len = hidden_states.shape[:2]

            if encoder_hidden_states is None:  # Cached Self Attn
                if self.cache_k is None:
                    self.cache_k = torch.zeros(
                        (batch_size, self.n_heads, self.max_seq_len, self.key_value_proj_dim)
                    ).to(hidden_states)
                if self.cache_v is None:
                    self.cache_v = torch.zeros(
                        (batch_size, self.n_heads, self.max_seq_len, self.key_value_proj_dim)
                    ).to(hidden_states)
                self.cache_k[:batch_size, :, start: start + seq_len, :] = self.reshape(
                    self.k(hidden_states) + apply_lora(hidden_states, self.lora_a_k, self.lora_b_k)
                )
                self.cache_v[:batch_size, :, start: start + seq_len, :] = self.reshape(
                    self.v(hidden_states) + apply_lora(hidden_states, self.lora_a_v, self.lora_b_v)
                )
                key_states = self.cache_k[:batch_size, :, : start + seq_len, :]
                value_states = self.cache_v[:batch_size, :, : start + seq_len, :]
            else:  # Cached Cross Attn
                if self.cache_k is None:
                    self.cache_k = self.reshape(
                        self.k(encoder_hidden_states) + apply_lora(encoder_hidden_states, self.lora_a_k, self.lora_b_k)
                    )
                if self.cache_v is None:
                    self.cache_v = self.reshape(
                        self.v(encoder_hidden_states) + apply_lora(encoder_hidden_states, self.lora_a_v, self.lora_b_v)
                    )
                key_states = self.cache_k
                value_states = self.cache_v
        return key_states, value_states

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor,
            position_bias: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            use_cache: bool = False,
            start_pos: int = None
    ):
        query_states = self.reshape(
            self.q(hidden_states) + apply_lora(hidden_states, self.lora_a_q, self.lora_b_q)
        )
        key_states, value_states = self.get_key_value_states(
            hidden_states, encoder_hidden_states, use_cache, start_pos)
        scores = torch.matmul(query_states, key_states.transpose(-1, -2))

        if position_bias is None:
            position_bias = self.get_position_bias(
                hidden_states.shape[1], use_cache, start_pos,
                encoder_seq_length=None if (
                        encoder_hidden_states is None
                ) else encoder_hidden_states.shape[1]
            )
        if attention_mask is not None:
            position_bias = position_bias + attention_mask.to(position_bias.device)

        scores = scores + position_bias.to(scores)
        attn_weights = torch.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = self.dropout.forward(attn_weights)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = torch.reshape(
            torch.transpose(
                attn_output, 1, 2
            ), shape=[hidden_states.shape[0], -1, self.inner_dim]
        )
        attn_output = self.o(attn_output) + apply_lora(attn_output, self.lora_a_o, self.lora_b_o)
        Outputs = collections.namedtuple('Outputs', ['hidden_states', 'position_bias'])
        return Outputs(hidden_states=attn_output, position_bias=position_bias)


class LoraParallelT5Feedforward(ParallelT5Feedforward):
    def __init__(self, config: LoraT5Config):
        super().__init__(config)
        self.config = config
        self.lora_a_wi_0 = None
        self.lora_b_wi_0 = None
        self.lora_a_wi_1 = None
        self.lora_b_wi_1 = None
        self.lora_a_wo = None
        self.lora_b_wo = None

    def init_weights(self):
        super().init_weights()
        self.lora_a_wi_0 = nn.Linear(
            self.config.d_model, self.config.r, bias=False
        ).type(self.config.lora_dtype)
        self.lora_b_wi_0 = ColumnParallelLinear(
            self.config.r, self.config.d_ff, bias=False, gather_output=False, init_method=init.zeros_
        ).type(self.config.lora_dtype)

        self.lora_a_wi_1 = nn.Linear(
            self.config.d_model, self.config.r, bias=False
        ).type(self.config.lora_dtype)
        self.lora_b_wi_1 = ColumnParallelLinear(
            self.config.r, self.config.d_ff, bias=False, gather_output=False, init_method=init.zeros_
        ).type(self.config.lora_dtype)

        self.lora_a_wo = RowParallelLinear(
            self.config.d_ff, self.config.r, bias=False, input_is_parallel=True, init_method=init.xavier_normal_
        ).type(self.config.lora_dtype)
        self.lora_b_wo = nn.Linear(
            self.config.r, self.config.d_model, bias=False
        ).type(self.config.lora_dtype)
        init.zeros_(self.lora_b_wo.weight)

    def forward(self, hidden_states):
        hidden_gelu = self.act(
            self.wi_0(hidden_states) + apply_lora(hidden_states, self.lora_a_wi_0, self.lora_b_wi_0)
        )
        hidden_linear = self.wi_1(hidden_states) + apply_lora(hidden_states, self.lora_a_wi_1, self.lora_b_wi_1)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states) + apply_lora(hidden_states, self.lora_a_wo, self.lora_b_wo)
        return hidden_states


# TODO
