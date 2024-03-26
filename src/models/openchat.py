from fairscale.nn.model_parallel.layers import RowParallelLinear

from src.models.mistral_hf import MistralHf
from src.models.modeling_args import OpenChatArgs


def get_lm_head_partition(local_rank: int, world_size: int, vocab_size: int) -> int:
    if world_size == 1:
        return vocab_size
    partition = vocab_size // world_size
    remained = vocab_size % world_size
    assert remained == 2
    return partition + remained if local_rank == world_size - 1 else partition


class OpenChat(MistralHf):
    def __init__(self, args: OpenChatArgs):
        super().__init__(args)

    def init_weights(self):
        self.model.init_weights()
        self.lm_head = RowParallelLinear(
            self.args.dim,
            self.args.vocab_size,
            bias=False,
            init_method=lambda x: x
        )

    def load(self, ckpt_dir: str, verbose: bool = True, num_added_tokens=2, **kwargs):
        super().load(ckpt_dir, verbose, num_added_tokens=num_added_tokens)
