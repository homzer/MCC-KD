import time

import torch


class SlimLogits:
    def __init__(self, logits: torch.Tensor = None, n=5):
        self.n = n
        self.vocab_size = None
        self.batch_size = None
        self.max_seq_len = None
        self.values = None
        self.indices = None
        if logits is not None:
            assert len(logits.shape) == 3
            self.batch_size = logits.shape[0]
            self.max_seq_len = logits.shape[1]
            self.vocab_size = logits.shape[2]
            self._set(logits)

    def _set(self, logits: torch.Tensor):
        (batch_size, seq_len, vocab_size) = logits.shape
        assert self.batch_size == batch_size
        assert self.vocab_size == vocab_size
        if logits.device.type == "cpu":
            logits = logits.float()  # topk is not implemented for half on cpu
        values, indices = torch.topk(logits, k=self.n)
        self.values = values.detach().cpu()
        self.indices = indices.detach().cpu()

    def extend(self, slim_logits: "SlimLogits"):
        """ Batch extend. """
        if self.vocab_size is None:
            self.vocab_size = slim_logits.vocab_size
        else:
            assert self.vocab_size == slim_logits.vocab_size
        if self.max_seq_len is None:
            self.max_seq_len = slim_logits.max_seq_len
        else:
            assert self.max_seq_len == slim_logits.max_seq_len

        self.values = slim_logits.values if (
                self.values is None
        ) else torch.cat([self.values, slim_logits.values], dim=0)
        self.indices = slim_logits.indices if (
                self.indices is None
        ) else torch.cat([self.indices, slim_logits.indices], dim=0)

    def __len__(self):
        if self.values is not None and self.indices is not None:
            return len(self.values)
        return 0

    def fetch(self, i: int) -> torch.Tensor:
        assert 0 <= i < len(self), "Index out of range error"
        value = self.values[i]  # [s, n]
        index = self.indices[i]  # [s, n]
        logits = torch.full(
            (self.max_seq_len, self.vocab_size),
            fill_value=torch.finfo(torch.get_default_dtype()).min
        )
        for j in range(self.max_seq_len):
            logits[j, index[j]] = value[j].to(logits)
        return logits  # [s, v]


class Timer:
    def __init__(self, total: int, episode: int = 1):
        self.total = total
        self.ticktock = 0
        self.last = None
        self.avg_time = 0
        self.episode = episode

    @staticmethod
    def format_clock(period):
        hour, minute, second = period // 3600, (period % 3600) // 60, period % 60
        return int(hour), int(minute), int(second)

    def step(self):
        if self.last is not None:
            period = time.time() - self.last
            self.avg_time = (self.avg_time * (self.ticktock - 1) + period) / self.ticktock
            h1, m1, s1 = self.format_clock(self.avg_time * (self.ticktock + 1))
            h2, m2, s2 = self.format_clock(self.avg_time * (self.total - self.ticktock))
            if self.ticktock % self.episode == 0:
                print(
                    f"STEP {self.ticktock}/{self.total} | USED: %02d:%02d:%02d | VAG %.2f s/it | "
                    f"ETA: %02d:%02d:%02d" % (h1, m1, s1, self.avg_time, h2, m2, s2)
                )
        self.last = time.time()
        self.ticktock += 1
        if self.ticktock == self.total:
            self.reset()

    def reset(self):
        self.ticktock = 0
        self.last = None
        self.avg_time = 0


class AverageMeter:
    def __init__(self):
        self.avg = 0
        self.step = 0

    def forward(self, x: int):
        """ Accumulate average computation """
        self.step += 1
        self.avg = self.avg + 1 / self.step * (x - self.avg)
        return self.avg

    def reset(self):
        self.avg = 0
        self.step = 0
