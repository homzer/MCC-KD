import torch
import torch.nn as nn
from transformers.activations import ACT2FN

from src.models.modeling import ModelForCausalLM, CausalLMOutputs
from src.models.modeling_args import GPT2Args
from src.utils import logits_normalize


class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x


class GPT2Attention(nn.Module):
    def __init__(self, args: GPT2Args):
        super().__init__()
        self.args = args
        max_positions = args.n_positions
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4))
        self.embed_dim = args.n_embd
        self.num_heads = args.n_head
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)
        self.attn_dropout = nn.Dropout(args.attn_pdrop)
        self.resid_dropout = nn.Dropout(args.resid_pdrop)

        self.cache_k = None
        self.cache_v = None

    def _attn(self, query, key, value, attention_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        attn_weights = attn_weights / torch.full(
            [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
        )

        # if only "normal" attention layer implements causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length].to(torch.bool)
        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = torch.softmax(attn_weights, dim=-1)

        # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def forward(
            self,
            x,
            attention_mask=None,
            start_pos=0,
            use_cache=False,
    ):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.c_attn(x).split(self.split_size, dim=2)
        xq = xq.view(bsz, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        xk = xk.view(bsz, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        xv = xv.view(bsz, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if use_cache:
            if self.cache_k is None:
                self.cache_k = torch.zeros((bsz, self.num_heads, self.args.max_seq_len, self.head_dim)).to(xk)
            if self.cache_v is None:
                self.cache_v = torch.zeros((bsz, self.num_heads, self.args.max_seq_len, self.head_dim)).to(xv)
            self.cache_k[:bsz, :, start_pos: start_pos + seq_len, :] = xk.clone()
            self.cache_v[:bsz, :, start_pos: start_pos + seq_len, :] = xv.clone()
            xk = self.cache_k[:bsz, :, : start_pos + seq_len, :]
            xv = self.cache_v[:bsz, :, : start_pos + seq_len, :]

        attn_output, attn_weights = self._attn(xq, xk, xv, attention_mask)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(bsz, seq_len, self.num_heads * self.head_dim)

        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output  # a, present, (attentions)

    def flush(self):
        self.cache_k = None
        self.cache_v = None


class GPT2MLP(nn.Module):
    def __init__(self, args: GPT2Args):
        super().__init__()
        intermediate_size = args.n_embd * 4
        self.c_fc = Conv1D(intermediate_size, args.n_embd)
        self.c_proj = Conv1D(args.n_embd, intermediate_size)
        self.act = ACT2FN[args.activation_function]
        self.dropout = nn.Dropout(args.resid_pdrop)

    def forward(self, x):
        x = self.c_proj(self.act(self.c_fc(x)))
        x = self.dropout(x)
        return x


class GPT2Block(nn.Module):
    def __init__(self, args: GPT2Args):
        super().__init__()
        self.ln_1 = nn.LayerNorm(args.n_embd, eps=args.layer_norm_epsilon)
        self.attn = GPT2Attention(args)
        self.ln_2 = nn.LayerNorm(args.n_embd, eps=args.layer_norm_epsilon)
        self.mlp = GPT2MLP(args)

    def forward(self, x, start_pos: int, masks, use_cache: bool):
        x = self.attn.forward(self.ln_1(x), masks, start_pos, use_cache) + x
        x = self.mlp.forward(self.ln_2(x)) + x
        return x


class GPT2(ModelForCausalLM):
    def __init__(self, args: GPT2Args):
        super().__init__()
        self.args = args
        self.embed_dim = args.n_embd

        self.wte = nn.Embedding(args.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(args.n_positions, self.embed_dim)

        self.drop = nn.Dropout(args.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(args) for _ in range(args.n_layer)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=args.layer_norm_epsilon)
        self.lm_head = None

    def init_weights(self):
        """ Obviate the need for GPT2. """
        pass

    def _init_lm_head(self):
        if self.lm_head is None:
            self.lm_head = nn.Linear(self.args.n_embd, self.args.vocab_size, bias=False)
            self.lm_head.weight = self.wte.weight

    def forward(
            self,
            tokens: torch.Tensor,
            start_pos: int = 0,
            use_cache: bool = False
    ):
        tokens = tokens.to(self.device())
        _bsz, seqlen = tokens.shape
        position_ids = torch.arange(start_pos, seqlen + start_pos, dtype=torch.long, device=self.device())
        position_ids = position_ids.unsqueeze(0).view(-1, seqlen)

        h = self.wte(tokens) + self.wpe(position_ids)
        h = self.drop(h)

        mask = None
        if seqlen > 1:
            mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=start_pos + 1).type_as(h)

        for layer in self.h:
            h = layer(h, start_pos, mask, use_cache)

        h = self.ln_f(h)
        self._init_lm_head()
        logits = self.lm_head(h)
        logits = logits_normalize(logits)
        return CausalLMOutputs(
            logits=logits.float(),
            hidden_states=h.float()
        )

    def flush(self):
        """ Clean cache in `LlamaAttention` module """
        for i in range(self.args.n_layer):
            self.h[i].attn.flush()
